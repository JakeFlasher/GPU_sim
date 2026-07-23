// slug: diffusion_resnet_time_block_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// Diffusion ResNet block with FiLM-style time embedding and 3x3
// same-padding convolutions. Matches references/diffusion.py::_resnet_time_block
// (plan-row 60: "time embedding + conv/norm/SiLU").
//
// 5-kernel pipeline:
//   K1: time_bias[b, c] = sum_t t[b, t] * W_time[t, c]
//   K2: h_pre1 = conv3x3_same(x, W_conv1)                    (BF16 -> FP32)
//   K3: h1     = silu(groupnorm_1g(h_pre1)) + time_bias[b, c]
//   K4: h_pre2 = conv3x3_same(h1, W_conv2)
//   K5: out    = x + silu(groupnorm_1g(h_pre2))               (residual)

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define RESNET_EPS 1e-5f

static __device__ __forceinline__ float silu_act(float v) {
    return v / (1.0f + __expf(-v));
}

// K1: time_bias projection. grid (B, C), threads reduce over T.
extern "C" __global__ void diffusion_resnet_time_bias_kernel(
    const __nv_bfloat16 *__restrict__ t,
    const __nv_bfloat16 *__restrict__ W_time,
    float *__restrict__ time_bias,
    int32_t T, int32_t C)
{
    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    extern __shared__ float smem[];
    float acc = 0.0f;
    for (int tt = tid; tt < T; tt += lanes) {
        acc += __bfloat162float(t[b * T + tt]) * __bfloat162float(W_time[tt * C + c]);
    }
    smem[tid] = acc;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) smem[tid] += smem[tid + st];
        __syncthreads();
    }
    if (tid == 0) time_bias[b * C + c] = smem[0];
}

// K2 / K4: 3x3 same-padding conv. grid (B, C_out), threads over H*W.
// Input is BF16 in K2 (the original x) and FP32 in K4 (h1). To handle
// both, we provide two specializations.
extern "C" __global__ void diffusion_resnet_conv3x3_bf16in_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ W_conv,
    float *__restrict__ out,
    int32_t C, int32_t H, int32_t W)
{
    const int b  = blockIdx.x;
    const int co = blockIdx.y;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    const int N = H * W;
    for (int idx = tid; idx < N; idx += lanes) {
        const int h = idx / W;
        const int w = idx % W;
        float acc = 0.0f;
        for (int ci = 0; ci < C; ++ci) {
            const size_t x_chan_base = ((size_t)b * C + ci) * N;
            for (int kh = 0; kh < 3; ++kh) {
                int ih = h + kh - 1;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < 3; ++kw) {
                    int iw = w + kw - 1;
                    if (iw < 0 || iw >= W) continue;
                    float xv = __bfloat162float(x[x_chan_base + ih * W + iw]);
                    float wv = __bfloat162float(W_conv[(((size_t)co * C + ci) * 3 + kh) * 3 + kw]);
                    acc += xv * wv;
                }
            }
        }
        out[((size_t)b * C + co) * N + idx] = acc;
    }
}

extern "C" __global__ void diffusion_resnet_conv3x3_fp32in_kernel(
    const float *__restrict__ x,
    const __nv_bfloat16 *__restrict__ W_conv,
    float *__restrict__ out,
    int32_t C, int32_t H, int32_t W)
{
    const int b  = blockIdx.x;
    const int co = blockIdx.y;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    const int N = H * W;
    for (int idx = tid; idx < N; idx += lanes) {
        const int h = idx / W;
        const int w = idx % W;
        float acc = 0.0f;
        for (int ci = 0; ci < C; ++ci) {
            const size_t x_chan_base = ((size_t)b * C + ci) * N;
            for (int kh = 0; kh < 3; ++kh) {
                int ih = h + kh - 1;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < 3; ++kw) {
                    int iw = w + kw - 1;
                    if (iw < 0 || iw >= W) continue;
                    float xv = x[x_chan_base + ih * W + iw];
                    float wv = __bfloat162float(W_conv[(((size_t)co * C + ci) * 3 + kh) * 3 + kw]);
                    acc += xv * wv;
                }
            }
        }
        out[((size_t)b * C + co) * N + idx] = acc;
    }
}

// K3: GroupNorm (1 group over C*H*W per batch) + SiLU + time_bias.
// Grid (B), threads cooperate over the full (C*H*W) volume.
extern "C" __global__ void diffusion_resnet_gn_silu_bias_kernel(
    const float *__restrict__ h_pre,
    const float *__restrict__ time_bias,   // [B, C]
    float *__restrict__ h_out,             // [B, C, H, W]
    int32_t C, int32_t H, int32_t W)
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    const int N = C * H * W;
    extern __shared__ float smem[];
    float *s_sum = smem;
    float *s_ss  = smem + lanes;
    const size_t base = (size_t)b * N;

    float m = 0.0f;
    for (int i = tid; i < N; i += lanes) m += h_pre[base + i];
    s_sum[tid] = m;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_sum[tid] += s_sum[tid + st];
        __syncthreads();
    }
    const float mean = s_sum[0] / (float)N;
    float v = 0.0f;
    for (int i = tid; i < N; i += lanes) {
        float t = h_pre[base + i] - mean;
        v += t * t;
    }
    s_ss[tid] = v;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_ss[tid] += s_ss[tid + st];
        __syncthreads();
    }
    const float inv_std = rsqrtf(s_ss[0] / (float)N + RESNET_EPS);
    const int HW = H * W;
    for (int i = tid; i < N; i += lanes) {
        int c = i / HW;
        float v0 = (h_pre[base + i] - mean) * inv_std;
        h_out[base + i] = silu_act(v0) + time_bias[b * C + c];
    }
}

// K5: GroupNorm + SiLU + residual cast to BF16.
extern "C" __global__ void diffusion_resnet_gn_silu_residual_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ h_pre,
    __nv_bfloat16 *__restrict__ out,
    int32_t C, int32_t H, int32_t W)
{
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    const int N = C * H * W;
    extern __shared__ float smem[];
    float *s_sum = smem;
    float *s_ss  = smem + lanes;
    const size_t base = (size_t)b * N;

    float m = 0.0f;
    for (int i = tid; i < N; i += lanes) m += h_pre[base + i];
    s_sum[tid] = m;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_sum[tid] += s_sum[tid + st];
        __syncthreads();
    }
    const float mean = s_sum[0] / (float)N;
    float v = 0.0f;
    for (int i = tid; i < N; i += lanes) {
        float t = h_pre[base + i] - mean;
        v += t * t;
    }
    s_ss[tid] = v;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_ss[tid] += s_ss[tid + st];
        __syncthreads();
    }
    const float inv_std = rsqrtf(s_ss[0] / (float)N + RESNET_EPS);
    for (int i = tid; i < N; i += lanes) {
        float xv = __bfloat162float(x[base + i]);
        float h2 = silu_act((h_pre[base + i] - mean) * inv_std);
        out[base + i] = __float2bfloat16(xv + h2);
    }
}

extern "C" void diffusion_resnet_time_block_bf16_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_t,
    const __nv_bfloat16 *d_W_time,
    const __nv_bfloat16 *d_W_conv1,
    const __nv_bfloat16 *d_W_conv2,
    float *d_time_bias,
    float *d_h_pre1,
    float *d_h1,
    float *d_h_pre2,
    __nv_bfloat16 *d_out,
    int B, int C, int H, int W, int T,
    cudaStream_t stream)
{
    int threads = 256;
    size_t shmem_2 = 2 * (size_t)threads * sizeof(float);
    size_t shmem_1 = (size_t)threads * sizeof(float);

    // K1: time bias projection.
    {
        dim3 grid_bc(B, C);
        dim3 block_t(threads);
        diffusion_resnet_time_bias_kernel<<<grid_bc, block_t, shmem_1, stream>>>(
            d_t, d_W_time, d_time_bias, T, C);
    }
    // K2: 3x3 conv on BF16 x -> h_pre1 (FP32).
    {
        dim3 grid_bc(B, C);
        dim3 block_t(threads);
        diffusion_resnet_conv3x3_bf16in_kernel<<<grid_bc, block_t, 0, stream>>>(
            d_x, d_W_conv1, d_h_pre1, C, H, W);
    }
    // K3: GN + SiLU + time_bias on h_pre1 -> h1.
    {
        dim3 grid_b(B);
        dim3 block_t(threads);
        diffusion_resnet_gn_silu_bias_kernel<<<grid_b, block_t, shmem_2, stream>>>(
            d_h_pre1, d_time_bias, d_h1, C, H, W);
    }
    // K4: 3x3 conv on FP32 h1 -> h_pre2.
    {
        dim3 grid_bc(B, C);
        dim3 block_t(threads);
        diffusion_resnet_conv3x3_fp32in_kernel<<<grid_bc, block_t, 0, stream>>>(
            d_h1, d_W_conv2, d_h_pre2, C, H, W);
    }
    // K5: GN + SiLU + residual.
    {
        dim3 grid_b(B);
        dim3 block_t(threads);
        diffusion_resnet_gn_silu_residual_kernel<<<grid_b, block_t, shmem_2, stream>>>(
            d_x, d_h_pre2, d_out, C, H, W);
    }
}
