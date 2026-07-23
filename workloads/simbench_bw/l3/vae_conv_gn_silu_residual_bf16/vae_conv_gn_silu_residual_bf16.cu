// slug: vae_conv_gn_silu_residual_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// VAE-style fused block: 3x3 same-padding convolution -> GroupNorm ->
// SiLU -> residual-add. BF16 I/O, FP32 accumulate, FP32 intermediate
// conv output, BF16 final output. Matches the Python reference at
// `references/conv.py::build_reference` bit-path-for-bit-path under
// the `(BF16, tree_sum, conv)` tolerance triplet.
//
// The row is expressed as three sequential launches sharing device
// buffers:
//
//   1. conv_3x3_kernel          — FP32 accumulate, emits float
//                                 `conv_out[N][C][H][W]` (intermediate).
//   2. gn_reduce_kernel         — reduces `conv_out` to per-(batch,
//                                 group) FP32 `mean[N][G]` and
//                                 `var[N][G]` (one block per
//                                 `(batch, group)`, block reduction).
//   3. gn_silu_residual_kernel  — normalizes `conv_out`, scales by
//                                 gamma + beta, applies SiLU, adds
//                                 residual `X`, casts to BF16 `Y`.
//
// Shape file launch_config captures the conv kernel wrapper
// (`grid = (batch, channels, height, 1), block = (width, 1, 1)`) —
// that is the primary numerical work. GN reduction and finalization
// use their own grids derived internally; only the conv launch is
// exposed in the shape file because `trace_size_estimator.py`'s
// byte-prediction model is calibrated against one launch per kernel
// row.
//
// Group count: `channels / 32` when `channels >= 32`, else `1` group
// covering all channels. Epsilon for GN = 1e-5. GN is inclusive (both
// the first and last channels of the group's reduction cover every
// `(h, w)` spatial position).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void vae_conv_gn_silu_residual_bf16_kernel(
    const __nv_bfloat16 *__restrict__ X,
    const __nv_bfloat16 *__restrict__ Wt,
    const __nv_bfloat16 *__restrict__ bias,
    float               *__restrict__ conv_out,
    int32_t N, int32_t C, int32_t H, int32_t Width)
{
    const int n = blockIdx.x;
    const int cout = blockIdx.y;
    const int y = blockIdx.z;
    const int x = threadIdx.x;
    if (x >= Width) return;

    float acc = __bfloat162float(bias[cout]);
    for (int cin = 0; cin < C; ++cin) {
        for (int ky = 0; ky < 3; ++ky) {
            const int iy = y + ky - 1;
            if (iy < 0 || iy >= H) continue;
            for (int kx = 0; kx < 3; ++kx) {
                const int ix = x + kx - 1;
                if (ix < 0 || ix >= Width) continue;
                const float xv = __bfloat162float(X[((n * C + cin) * H + iy) * Width + ix]);
                const float wv = __bfloat162float(Wt[((cout * C + cin) * 3 + ky) * 3 + kx]);
                acc += xv * wv;
            }
        }
    }
    conv_out[((n * C + cout) * H + y) * Width + x] = acc;
}


extern "C" __global__ void vae_gn_reduce_kernel(
    const float *__restrict__ conv_out,
    float       *__restrict__ mean_out,    // [N, num_groups]
    float       *__restrict__ var_out,     // [N, num_groups]
    int32_t N, int32_t C, int32_t H, int32_t Width,
    int32_t group_size, int32_t num_groups)
{
    const int n = blockIdx.x;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;
    const int block_sz = blockDim.x;
    const int c_start = g * group_size;
    const int spatial = H * Width;
    const int total = group_size * spatial;

    extern __shared__ float sdata[];
    float *s_sum   = sdata;
    float *s_sqsum = sdata + block_sz;

    float local_sum = 0.f, local_sqsum = 0.f;
    for (int i = tid; i < total; i += block_sz) {
        const int c_off = i / spatial;
        const int sp    = i % spatial;
        const int h_    = sp / Width;
        const int w_    = sp % Width;
        const int c     = c_start + c_off;
        const float v = conv_out[((n * C + c) * H + h_) * Width + w_];
        local_sum   += v;
        local_sqsum += v * v;
    }
    s_sum[tid]   = local_sum;
    s_sqsum[tid] = local_sqsum;
    __syncthreads();

    for (int s = block_sz >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid]   += s_sum[tid + s];
            s_sqsum[tid] += s_sqsum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float inv_total = 1.0f / (float)total;
        const float mean = s_sum[0] * inv_total;
        float var = s_sqsum[0] * inv_total - mean * mean;
        if (var < 0.f) var = 0.f;
        mean_out[n * num_groups + g] = mean;
        var_out[n * num_groups + g]  = var;
    }
}


extern "C" __global__ void vae_gn_silu_residual_kernel(
    const float         *__restrict__ conv_out,
    const float         *__restrict__ mean,
    const float         *__restrict__ var,
    const __nv_bfloat16 *__restrict__ gamma,
    const __nv_bfloat16 *__restrict__ beta,
    const __nv_bfloat16 *__restrict__ X,
    __nv_bfloat16       *__restrict__ Y,
    int32_t N, int32_t C, int32_t H, int32_t Width,
    int32_t group_size, int32_t num_groups)
{
    const int n = blockIdx.x;
    const int c = blockIdx.y;
    const int h_ = blockIdx.z;
    const int w_ = threadIdx.x;
    if (w_ >= Width) return;

    const int g = (group_size > 0) ? (c / group_size) : 0;
    const float m    = mean[n * num_groups + g];
    const float v    = var[n  * num_groups + g];
    const float rstd = rsqrtf(v + 1e-5f);
    const float gm   = __bfloat162float(gamma[c]);
    const float bt   = __bfloat162float(beta[c]);

    const int idx = ((n * C + c) * H + h_) * Width + w_;
    const float conv_val = conv_out[idx];
    const float normed = (conv_val - m) * rstd;
    const float scaled = normed * gm + bt;
    const float sig    = 1.0f / (1.0f + __expf(-scaled));
    const float silu   = scaled * sig;
    const float xv     = __bfloat162float(X[idx]);
    Y[idx] = __float2bfloat16(silu + xv);
}


extern "C" void vae_conv_gn_silu_residual_bf16_launch(
    const __nv_bfloat16 *d_X,
    const __nv_bfloat16 *d_W,
    const __nv_bfloat16 *d_b,
    const __nv_bfloat16 *d_gamma,
    const __nv_bfloat16 *d_beta,
    float               *d_conv_out,
    float               *d_mean,
    float               *d_var,
    __nv_bfloat16       *d_Y,
    int N, int C, int H, int Width,
    cudaStream_t stream)
{
    const int group_size = (C >= 32) ? 32 : C;
    const int num_groups = (group_size > 0) ? (C / group_size) : 1;

    // Pass 1: conv. Grid exposes one block per (batch, channel_out, y).
    // block.x = width. This is the launch_config recorded in the shape file.
    dim3 conv_grid(N, C, H);
    dim3 conv_block(Width, 1, 1);
    vae_conv_gn_silu_residual_bf16_kernel<<<conv_grid, conv_block, 0, stream>>>(
        d_X, d_W, d_b, d_conv_out, N, C, H, Width);

    // Pass 2: reduction. One block per (batch, group); 256 threads per block.
    const int reduce_threads = 256;
    dim3 reduce_grid(N, num_groups, 1);
    dim3 reduce_block(reduce_threads, 1, 1);
    size_t reduce_shmem = (size_t)(2 * reduce_threads) * sizeof(float);
    vae_gn_reduce_kernel<<<reduce_grid, reduce_block, reduce_shmem, stream>>>(
        d_conv_out, d_mean, d_var, N, C, H, Width, group_size, num_groups);

    // Pass 3: normalize + SiLU + residual + BF16 cast. Same grid as conv.
    vae_gn_silu_residual_kernel<<<conv_grid, conv_block, 0, stream>>>(
        d_conv_out, d_mean, d_var, d_gamma, d_beta, d_X, d_Y,
        N, C, H, Width, group_size, num_groups);
}
