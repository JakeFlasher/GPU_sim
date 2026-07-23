// slug: flash_attn_bwd_fp8
// tier: P
// arch_gate: sm_90a sm_120
//
// Scaled-dot-product attention BACKWARD pass (non-causal), FP8 E4M3
// storage for Q/K/V/dO/dQ. R39 repairs the R38 placeholder
// which used BF16 storage — the row's declared primary_precision is
// FP8_E4M3, so the tensors must live in FP8.
//
// Internal accumulation is FP32; cast at the boundary via the
// __nv_fp8_e4m3 struct's explicit float constructor + float operator.
// Tolerance: (FP8_E4M3, atomic_add, attention) = rtol = atol = 0.125
// (deterministic = "ND" routes to the atomic-add band; the wide band
// absorbs the FP8 quantization of Q/K/V/dO).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>


static __forceinline__ __device__ float fp8_to_f32(__nv_fp8_e4m3 v) {
    return float(v);
}


extern "C" __global__ void flash_attn_bwd_fp8_kernel(
    const __nv_fp8_e4m3 *__restrict__ q,
    const __nv_fp8_e4m3 *__restrict__ k,
    const __nv_fp8_e4m3 *__restrict__ v,
    const __nv_fp8_e4m3 *__restrict__ dO,
    __nv_fp8_e4m3 *__restrict__ dq,
    int32_t heads,
    int32_t seq,
    int32_t head_dim,
    float scale)
{
    const int qi = blockIdx.x;
    const int h  = blockIdx.y;
    const int ba = blockIdx.z;
    const int d  = threadIdx.x;
    if (d >= head_dim) return;

    extern __shared__ float shared_buf[];
    float *shared_P  = shared_buf;
    float *shared_ds = shared_buf + seq;

    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j < seq; ++j) {
            float s = 0.0f;
            for (int dk = 0; dk < head_dim; ++dk) {
                float qv = fp8_to_f32(q[(((ba * heads) + h) * seq + qi) * head_dim + dk]);
                float kv = fp8_to_f32(k[(((ba * heads) + h) * seq +  j) * head_dim + dk]);
                s += qv * kv;
            }
            shared_P[j] = s * scale;
            if (shared_P[j] > m) m = shared_P[j];
        }
        float z = 0.0f;
        for (int j = 0; j < seq; ++j) {
            shared_P[j] = __expf(shared_P[j] - m);
            z += shared_P[j];
        }
        if (z > 0.0f) for (int j = 0; j < seq; ++j) shared_P[j] /= z;

        float row_dot = 0.0f;
        for (int j = 0; j < seq; ++j) {
            float dp = 0.0f;
            for (int dk = 0; dk < head_dim; ++dk) {
                float dov = fp8_to_f32(dO[(((ba * heads) + h) * seq + qi) * head_dim + dk]);
                float vv  = fp8_to_f32(v[(((ba * heads) + h) * seq +  j) * head_dim + dk]);
                dp += dov * vv;
            }
            shared_ds[j] = dp;
            row_dot += shared_P[j] * dp;
        }
        for (int j = 0; j < seq; ++j) {
            shared_ds[j] = shared_P[j] * (shared_ds[j] - row_dot);
        }
    }
    __syncthreads();

    float acc = 0.0f;
    for (int j = 0; j < seq; ++j) {
        float kv = fp8_to_f32(k[(((ba * heads) + h) * seq + j) * head_dim + d]);
        acc += shared_ds[j] * kv;
    }
    acc *= scale;
    dq[(((ba * heads) + h) * seq + qi) * head_dim + d] = __nv_fp8_e4m3(acc);
}

extern "C" void flash_attn_bwd_fp8_launch(
    const __nv_fp8_e4m3 *d_q,
    const __nv_fp8_e4m3 *d_k,
    const __nv_fp8_e4m3 *d_v,
    const __nv_fp8_e4m3 *d_dO,
    __nv_fp8_e4m3 *d_dq,
    int batch, int heads, int seq, int head_dim,
    float scale, cudaStream_t stream)
{
    dim3 grid(seq, heads, batch);
    dim3 block(head_dim);
    size_t shmem = (size_t)(2 * seq) * sizeof(float);
    flash_attn_bwd_fp8_kernel<<<grid, block, shmem, stream>>>(
        d_q, d_k, d_v, d_dO, d_dq, heads, seq, head_dim, scale);
}
