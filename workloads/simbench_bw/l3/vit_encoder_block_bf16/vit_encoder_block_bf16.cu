// slug: vit_encoder_block_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// ViT encoder block: pre-norm self-attention + pre-norm 2-layer GELU
// MLP. Matches references/vision_encoder.py::_vit_encoder_block.
//
// 7-kernel pipeline chained through device-side workspace buffers.
// Each kernel is deliberately simple (one block per token or per
// (head, token)), optimizing for correctness parity with the Python
// reference at correctness-shape scale. Performance optimization
// (fused LN+matmul + warp-level reductions) is deferred to a later
// round.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define VIT_EPS 1e-5f

// --- K1: LayerNorm per token, output [B, S, hidden] in FP32 scratch ---
extern "C" __global__ void vit_encoder_ln_kernel(
    const __nv_bfloat16 *__restrict__ x,
    float *__restrict__ x_ln,
    int32_t S, int32_t hidden)
{
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    extern __shared__ float smem[];
    float *s_sum = smem;
    float *s_ss  = smem + lanes;

    float m = 0.0f;
    for (int d = tid; d < hidden; d += lanes) m += __bfloat162float(x[(b * S + s) * hidden + d]);
    s_sum[tid] = m;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_sum[tid] += s_sum[tid + st];
        __syncthreads();
    }
    const float mean = s_sum[0] / (float)hidden;
    float v = 0.0f;
    for (int d = tid; d < hidden; d += lanes) {
        float t = __bfloat162float(x[(b * S + s) * hidden + d]) - mean;
        v += t * t;
    }
    s_ss[tid] = v;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_ss[tid] += s_ss[tid + st];
        __syncthreads();
    }
    const float inv_std = rsqrtf(s_ss[0] / (float)hidden + VIT_EPS);
    for (int d = tid; d < hidden; d += lanes) {
        float t = __bfloat162float(x[(b * S + s) * hidden + d]) - mean;
        x_ln[(b * S + s) * hidden + d] = t * inv_std;
    }
}

// --- K2: QKV projection. out_qkv[B, S, 3*hidden] in FP32 ---
extern "C" __global__ void vit_encoder_qkv_proj_kernel(
    const float *__restrict__ x_ln,
    const __nv_bfloat16 *__restrict__ W_qkv,
    float *__restrict__ out_qkv,
    int32_t S, int32_t hidden)
{
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int c = threadIdx.x;
    const int three_h = 3 * hidden;
    for (int cc = c; cc < three_h; cc += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < hidden; ++d) {
            acc += x_ln[(b * S + s) * hidden + d] * __bfloat162float(W_qkv[d * three_h + cc]);
        }
        out_qkv[(b * S + s) * three_h + cc] = acc;
    }
}

// --- K3: SDPA per (batch, head, query-pos), writing attn[B, S, hidden] concatenated-by-head ---
extern "C" __global__ void vit_encoder_sdpa_kernel(
    const float *__restrict__ qkv,    // [B, S, 3*hidden]
    float *__restrict__ attn,          // [B, S, hidden] (concat by head)
    int32_t S, int32_t heads, int32_t head_dim, float scale)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int qi = blockIdx.z;
    const int d  = threadIdx.x;
    if (d >= head_dim) return;
    const int hidden = heads * head_dim;
    extern __shared__ float shared_scores[];
    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j < S; ++j) {
            float sc = 0.0f;
            for (int dk = 0; dk < head_dim; ++dk) {
                float qv = qkv[(b * S + qi) * 3 * hidden + 0 * hidden + h * head_dim + dk];
                float kv = qkv[(b * S + j)  * 3 * hidden + 1 * hidden + h * head_dim + dk];
                sc += qv * kv;
            }
            sc *= scale;
            shared_scores[j] = sc;
            if (sc > m) m = sc;
        }
        float z = 0.0f;
        for (int j = 0; j < S; ++j) { shared_scores[j] = __expf(shared_scores[j] - m); z += shared_scores[j]; }
        if (z > 0.0f) for (int j = 0; j < S; ++j) shared_scores[j] /= z;
    }
    __syncthreads();
    float acc = 0.0f;
    for (int j = 0; j < S; ++j) {
        float vv = qkv[(b * S + j) * 3 * hidden + 2 * hidden + h * head_dim + d];
        acc += shared_scores[j] * vv;
    }
    attn[(b * S + qi) * hidden + h * head_dim + d] = acc;
}

// --- K4: out projection + residual. h1 = x + attn @ W_out ---
extern "C" __global__ void vit_encoder_out_proj_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ attn,
    const __nv_bfloat16 *__restrict__ W_out,
    float *__restrict__ h1,
    int32_t S, int32_t hidden)
{
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int d = threadIdx.x;
    for (int dd = d; dd < hidden; dd += blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < hidden; ++c) {
            acc += attn[(b * S + s) * hidden + c] * __bfloat162float(W_out[c * hidden + dd]);
        }
        h1[(b * S + s) * hidden + dd] = __bfloat162float(x[(b * S + s) * hidden + dd]) + acc;
    }
}

// --- K5a: LayerNorm of h1 → h1_ln ---
extern "C" __global__ void vit_encoder_ln2_kernel(
    const float *__restrict__ h1,
    float *__restrict__ h1_ln,
    int32_t S, int32_t hidden)
{
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    extern __shared__ float smem[];
    float *s_sum = smem;
    float *s_ss  = smem + lanes;
    float m = 0.0f;
    for (int d = tid; d < hidden; d += lanes) m += h1[(b * S + s) * hidden + d];
    s_sum[tid] = m;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_sum[tid] += s_sum[tid + st];
        __syncthreads();
    }
    const float mean = s_sum[0] / (float)hidden;
    float v = 0.0f;
    for (int d = tid; d < hidden; d += lanes) {
        float t = h1[(b * S + s) * hidden + d] - mean;
        v += t * t;
    }
    s_ss[tid] = v;
    __syncthreads();
    for (int st = lanes / 2; st > 0; st >>= 1) {
        if (tid < st) s_ss[tid] += s_ss[tid + st];
        __syncthreads();
    }
    const float inv_std = rsqrtf(s_ss[0] / (float)hidden + VIT_EPS);
    for (int d = tid; d < hidden; d += lanes) {
        h1_ln[(b * S + s) * hidden + d] = (h1[(b * S + s) * hidden + d] - mean) * inv_std;
    }
}

// --- K6: Up projection + GELU ---
extern "C" __global__ void vit_encoder_up_gelu_kernel(
    const float *__restrict__ h1_ln,
    const __nv_bfloat16 *__restrict__ W_ff1,
    float *__restrict__ ff_inner,
    int32_t S, int32_t hidden, int32_t M)
{
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int k = threadIdx.x;
    for (int kk = k; kk < M; kk += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < hidden; ++d) {
            acc += h1_ln[(b * S + s) * hidden + d] * __bfloat162float(W_ff1[d * M + kk]);
        }
        // GELU tanh-approx.
        float inner = 0.7978845608f * (acc + 0.044715f * acc * acc * acc);
        ff_inner[(b * S + s) * M + kk] = 0.5f * acc * (1.0f + tanhf(inner));
    }
}

// --- K7: Down projection + residual ---
extern "C" __global__ void vit_encoder_down_proj_kernel(
    const float *__restrict__ h1,
    const float *__restrict__ ff_inner,
    const __nv_bfloat16 *__restrict__ W_ff2,
    __nv_bfloat16 *__restrict__ out,
    int32_t S, int32_t hidden, int32_t M)
{
    const int b = blockIdx.x;
    const int s = blockIdx.y;
    const int d = threadIdx.x;
    for (int dd = d; dd < hidden; dd += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < M; ++k) {
            acc += ff_inner[(b * S + s) * M + k] * __bfloat162float(W_ff2[k * hidden + dd]);
        }
        float h1_v = h1[(b * S + s) * hidden + dd];
        out[(b * S + s) * hidden + dd] = __float2bfloat16(h1_v + acc);
    }
}

extern "C" void vit_encoder_block_bf16_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_W_qkv,
    const __nv_bfloat16 *d_W_out,
    const __nv_bfloat16 *d_W_ff1,
    const __nv_bfloat16 *d_W_ff2,
    float *d_x_ln, float *d_qkv, float *d_attn, float *d_h1, float *d_h1_ln, float *d_ff_inner,
    __nv_bfloat16 *d_out,
    int B, int S, int heads, int head_dim, int M,
    cudaStream_t stream)
{
    const int hidden = heads * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);
    int threads = 256;
    dim3 grid_bs(B, S);
    dim3 block_t(threads);
    size_t shmem = 2 * threads * sizeof(float);

    vit_encoder_ln_kernel<<<grid_bs, block_t, shmem, stream>>>(d_x, d_x_ln, S, hidden);
    vit_encoder_qkv_proj_kernel<<<grid_bs, block_t, 0, stream>>>(d_x_ln, d_W_qkv, d_qkv, S, hidden);
    {
        dim3 grid_bhs(B, heads, S);
        dim3 block_d(head_dim);
        size_t shmem_sdpa = (size_t)S * sizeof(float);
        vit_encoder_sdpa_kernel<<<grid_bhs, block_d, shmem_sdpa, stream>>>(d_qkv, d_attn, S, heads, head_dim, scale);
    }
    vit_encoder_out_proj_kernel<<<grid_bs, block_t, 0, stream>>>(d_x, d_attn, d_W_out, d_h1, S, hidden);
    vit_encoder_ln2_kernel<<<grid_bs, block_t, shmem, stream>>>(d_h1, d_h1_ln, S, hidden);
    vit_encoder_up_gelu_kernel<<<grid_bs, block_t, 0, stream>>>(d_h1_ln, d_W_ff1, d_ff_inner, S, hidden, M);
    vit_encoder_down_proj_kernel<<<grid_bs, block_t, 0, stream>>>(d_h1, d_ff_inner, d_W_ff2, d_out, S, hidden, M);
}
