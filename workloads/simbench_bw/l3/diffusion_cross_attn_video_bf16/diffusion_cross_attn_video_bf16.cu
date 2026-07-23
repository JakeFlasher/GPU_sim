// slug: diffusion_cross_attn_video_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// Text-conditioned video cross attention. Q from video tokens
// `[batch, heads, seq_video, head_dim]`; K / V from text
// `[batch, heads, seq_text, head_dim]`. Standard scaled dot-product
// cross attention with row-wise softmax over the text sequence.
// Output `[batch, heads, seq_video, head_dim]`. Matches
// references/diffusion.py::_cross_attn_video.
//
// Grid: one block per (batch, head, seq_video); head_dim threads per
// block. Thread 0 computes scores + stable softmax; all threads
// accumulate the V-weighted output.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

extern "C" __global__ void diffusion_cross_attn_video_bf16_kernel(
    const __nv_bfloat16 *__restrict__ Q,  // [B, H, Sv, D]
    const __nv_bfloat16 *__restrict__ K,  // [B, H, St, D]
    const __nv_bfloat16 *__restrict__ V,  // [B, H, St, D]
    __nv_bfloat16 *__restrict__ out,       // [B, H, Sv, D]
    int32_t H, int32_t Sv, int32_t St, int32_t D, float scale)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int i = blockIdx.z;
    const int d = threadIdx.x;
    if (d >= D) return;

    extern __shared__ float shared_probs[];  // [St]

    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j < St; ++j) {
            float s = 0.0f;
            for (int dk = 0; dk < D; ++dk) {
                float qv = __bfloat162float(Q[(((b * H) + h) * Sv + i) * D + dk]);
                float kv = __bfloat162float(K[(((b * H) + h) * St + j) * D + dk]);
                s += qv * kv;
            }
            s *= scale;
            shared_probs[j] = s;
            if (s > m) m = s;
        }
        float z = 0.0f;
        for (int j = 0; j < St; ++j) { shared_probs[j] = __expf(shared_probs[j] - m); z += shared_probs[j]; }
        if (z > 0.0f) for (int j = 0; j < St; ++j) shared_probs[j] /= z;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int j = 0; j < St; ++j) {
        float vv = __bfloat162float(V[(((b * H) + h) * St + j) * D + d]);
        acc += shared_probs[j] * vv;
    }
    out[(((b * H) + h) * Sv + i) * D + d] = __float2bfloat16(acc);
}

extern "C" void diffusion_cross_attn_video_bf16_launch(
    const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K, const __nv_bfloat16 *d_V,
    __nv_bfloat16 *d_out,
    int B, int H, int Sv, int St, int D, float scale, cudaStream_t stream)
{
    dim3 grid(B, H, Sv);
    dim3 block(D);
    size_t shmem = (size_t)St * sizeof(float);
    diffusion_cross_attn_video_bf16_kernel<<<grid, block, shmem, stream>>>(
        d_Q, d_K, d_V, d_out, H, Sv, St, D, scale);
}
