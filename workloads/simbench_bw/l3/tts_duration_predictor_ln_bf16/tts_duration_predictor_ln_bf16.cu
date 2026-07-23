// slug: tts_duration_predictor_ln_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// TTS duration predictor: Conv1d(K=3, causal) -> +b1 -> SiLU ->
// LayerNorm(over H) -> Linear(H->1) -> exp.
// Matches references/tts.py::_duration_predictor (plan-row 67 contract:
// "duration predictor conv + LN").
//
// 2-kernel pipeline:
//   K1: per (b, s, h) hidden1[b, s, h] = silu(b1[h] +
//       sum_{k=0..K-1, c} X[b, s-k, c] * W_conv[h, c, k]) (zero-pad)
//   K2: per (b, s) consume hidden1, compute LN moments, project + exp.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define TTS_LN_EPS 1e-5f
#define DURATION_CONV_K 3

static __device__ __forceinline__ float silu_act(float v) {
    return v / (1.0f + __expf(-v));
}

extern "C" __global__ void tts_duration_conv_silu_kernel(
    const __nv_bfloat16 *__restrict__ X,
    const __nv_bfloat16 *__restrict__ W_conv,
    const __nv_bfloat16 *__restrict__ b1,
    float *__restrict__ hidden1,
    int32_t S, int32_t H)
{
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    const size_t out_base = ((size_t)b * S + s) * H;
    for (int h = tid; h < H; h += lanes) {
        float acc = __bfloat162float(b1[h]);
        for (int k = 0; k < DURATION_CONV_K; ++k) {
            int tau = s - k;
            if (tau < 0) continue;
            const size_t row_base = ((size_t)b * S + tau) * H;
            for (int c = 0; c < H; ++c) {
                float x_v = __bfloat162float(X[row_base + c]);
                float w_v = __bfloat162float(W_conv[((size_t)h * H + c) * DURATION_CONV_K + k]);
                acc += x_v * w_v;
            }
        }
        hidden1[out_base + h] = silu_act(acc);
    }
}

extern "C" __global__ void tts_duration_ln_proj_exp_kernel(
    const float *__restrict__ hidden1,
    const __nv_bfloat16 *__restrict__ gamma,
    const __nv_bfloat16 *__restrict__ beta,
    const __nv_bfloat16 *__restrict__ W2,
    const __nv_bfloat16 *__restrict__ b2_vec,
    __nv_bfloat16 *__restrict__ out,
    int32_t S, int32_t H)
{
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    extern __shared__ float smem[];
    float *s_sum = smem;
    float *s_ss  = smem + lanes;
    float *s_dot = smem + 2 * lanes;
    const size_t row_base = ((size_t)b * S + s) * H;

    float m = 0.0f;
    for (int h = tid; h < H; h += lanes) m += hidden1[row_base + h];
    s_sum[tid] = m;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_sum[tid] += s_sum[tid + st];
        __syncthreads();
    }
    const float mean = s_sum[0] / (float)H;
    float v = 0.0f;
    for (int h = tid; h < H; h += lanes) {
        float t = hidden1[row_base + h] - mean;
        v += t * t;
    }
    s_ss[tid] = v;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_ss[tid] += s_ss[tid + st];
        __syncthreads();
    }
    const float rstd = rsqrtf(s_ss[0] / (float)H + TTS_LN_EPS);
    float local_dot = 0.0f;
    for (int h = tid; h < H; h += lanes) {
        float normed = (hidden1[row_base + h] - mean) * rstd;
        float scaled = normed * __bfloat162float(gamma[h]) + __bfloat162float(beta[h]);
        local_dot += scaled * __bfloat162float(W2[h]);
    }
    s_dot[tid] = local_dot;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_dot[tid] += s_dot[tid + st];
        __syncthreads();
    }
    if (tid == 0) {
        float dot = s_dot[0] + __bfloat162float(b2_vec[0]);
        out[b * S + s] = __float2bfloat16(__expf(dot));
    }
}

extern "C" void tts_duration_predictor_ln_bf16_launch(
    const __nv_bfloat16 *d_X,
    const __nv_bfloat16 *d_W_conv, const __nv_bfloat16 *d_b1,
    const __nv_bfloat16 *d_gamma, const __nv_bfloat16 *d_beta,
    const __nv_bfloat16 *d_W2, const __nv_bfloat16 *d_b2_vec,
    float *d_hidden1,
    __nv_bfloat16 *d_out,
    int B, int S, int H,
    cudaStream_t stream)
{
    int threads = 256;
    dim3 grid(B, S);
    dim3 block_t(threads);
    tts_duration_conv_silu_kernel<<<grid, block_t, 0, stream>>>(
        d_X, d_W_conv, d_b1, d_hidden1, S, H);
    size_t shmem = 3 * (size_t)threads * sizeof(float);
    tts_duration_ln_proj_exp_kernel<<<grid, block_t, shmem, stream>>>(
        d_hidden1, d_gamma, d_beta, d_W2, d_b2_vec, d_out, S, H);
}
