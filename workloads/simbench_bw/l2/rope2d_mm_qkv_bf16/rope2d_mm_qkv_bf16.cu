// slug: rope2d_mm_qkv_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// 2D-RoPE applied to the fused QKV tensor. Tokens are placed on a
// [grid_h, grid_w] spatial grid (row-major by token index); each
// token's head_dim is split into two halves:
// - The first half rotates under the h-coordinate theta schedule.
// - The second half rotates under the w-coordinate theta schedule.
// Matches references/rope.py for slug == "rope2d_mm_qkv_bf16".
//
// Grid: one block per (token, head). Block: head_dim/4 threads — each
// thread rotates two adjacent-pair slots, one in the h-half and one
// in the w-half.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void rope2d_mm_qkv_bf16_kernel(
    __nv_bfloat16 *__restrict__ qkv,          // [tokens, heads, head_dim]
    const __nv_bfloat16 *__restrict__ cos_h,   // [grid_h, head_dim/4]
    const __nv_bfloat16 *__restrict__ sin_h,   // [grid_h, head_dim/4]
    const __nv_bfloat16 *__restrict__ cos_w,   // [grid_w, head_dim/4]
    const __nv_bfloat16 *__restrict__ sin_w,   // [grid_w, head_dim/4]
    int32_t heads,
    int32_t head_dim,
    int32_t grid_h,
    int32_t grid_w)
{
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int k = threadIdx.x;
    const int quarter = head_dim / 4;
    if (k >= quarter) return;
    const int half = head_dim / 2;
    const int p_h = token / grid_w;
    const int p_w = token % grid_w;
    (void)grid_h;

    // h-half rotation: slots [2k, 2k+1].
    {
        int base = ((token * heads) + head) * head_dim + 2 * k;
        float r_even = __bfloat162float(qkv[base]);
        float r_odd  = __bfloat162float(qkv[base + 1]);
        float c = __bfloat162float(cos_h[p_h * quarter + k]);
        float s = __bfloat162float(sin_h[p_h * quarter + k]);
        qkv[base]     = __float2bfloat16(r_even * c - r_odd * s);
        qkv[base + 1] = __float2bfloat16(r_even * s + r_odd * c);
    }
    // w-half rotation: slots [half + 2k, half + 2k + 1].
    {
        int base = ((token * heads) + head) * head_dim + half + 2 * k;
        float r_even = __bfloat162float(qkv[base]);
        float r_odd  = __bfloat162float(qkv[base + 1]);
        float c = __bfloat162float(cos_w[p_w * quarter + k]);
        float s = __bfloat162float(sin_w[p_w * quarter + k]);
        qkv[base]     = __float2bfloat16(r_even * c - r_odd * s);
        qkv[base + 1] = __float2bfloat16(r_even * s + r_odd * c);
    }
}

extern "C" void rope2d_mm_qkv_bf16_launch(
    __nv_bfloat16 *d_qkv,
    const __nv_bfloat16 *d_cos_h, const __nv_bfloat16 *d_sin_h,
    const __nv_bfloat16 *d_cos_w, const __nv_bfloat16 *d_sin_w,
    int tokens, int heads, int head_dim, int grid_h, int grid_w,
    cudaStream_t stream)
{
    dim3 grid(tokens, heads);
    dim3 block(head_dim / 4);
    rope2d_mm_qkv_bf16_kernel<<<grid, block, 0, stream>>>(
        d_qkv, d_cos_h, d_sin_h, d_cos_w, d_sin_w, heads, head_dim, grid_h, grid_w);
}
