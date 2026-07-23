// slug: sam_mask_decoder_cross_attn_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// SAM mask-decoder cross attention. Q from mask tokens
// `[batch, heads, seq_mask, head_dim]`; K / V from image tokens
// `[batch, heads, seq_image, head_dim]`. Row-wise softmax over image
// sequence. Matches references/vision_encoder.py::_sam_cross_attn.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

extern "C" __global__ void sam_mask_decoder_cross_attn_bf16_kernel(
    const __nv_bfloat16 *__restrict__ Q,
    const __nv_bfloat16 *__restrict__ K,
    const __nv_bfloat16 *__restrict__ V,
    __nv_bfloat16 *__restrict__ out,
    int32_t H, int32_t Sm, int32_t Si, int32_t D, float scale)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int i = blockIdx.z;
    const int d = threadIdx.x;
    if (d >= D) return;
    extern __shared__ float shared_probs[];
    if (d == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j < Si; ++j) {
            float s = 0.0f;
            for (int dk = 0; dk < D; ++dk) {
                float qv = __bfloat162float(Q[(((b * H) + h) * Sm + i) * D + dk]);
                float kv = __bfloat162float(K[(((b * H) + h) * Si + j) * D + dk]);
                s += qv * kv;
            }
            s *= scale;
            shared_probs[j] = s;
            if (s > m) m = s;
        }
        float z = 0.0f;
        for (int j = 0; j < Si; ++j) { shared_probs[j] = __expf(shared_probs[j] - m); z += shared_probs[j]; }
        if (z > 0.0f) for (int j = 0; j < Si; ++j) shared_probs[j] /= z;
    }
    __syncthreads();
    float acc = 0.0f;
    for (int j = 0; j < Si; ++j) {
        float vv = __bfloat162float(V[(((b * H) + h) * Si + j) * D + d]);
        acc += shared_probs[j] * vv;
    }
    out[(((b * H) + h) * Sm + i) * D + d] = __float2bfloat16(acc);
}

extern "C" void sam_mask_decoder_cross_attn_bf16_launch(
    const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K, const __nv_bfloat16 *d_V,
    __nv_bfloat16 *d_out,
    int B, int H, int Sm, int Si, int D, float scale, cudaStream_t stream)
{
    dim3 grid(B, H, Sm);
    dim3 block(D);
    size_t shmem = (size_t)Si * sizeof(float);
    sam_mask_decoder_cross_attn_bf16_kernel<<<grid, block, shmem, stream>>>(
        d_Q, d_K, d_V, d_out, H, Sm, Si, D, scale);
}
