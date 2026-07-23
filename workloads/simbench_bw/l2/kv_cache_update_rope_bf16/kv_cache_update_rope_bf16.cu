// slug: kv_cache_update_rope_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// In-place KV-cache update with RoPE rotation. Applies RoPE to each
// position `p in [0, context)` of the seeded K cache
// `[batch, kv_heads, context, head_dim]` and emits the rotated cache.
// Matches references/paged_attention.py::build_reference for slug
// "kv_cache_update_rope_bf16" (R39 repair; R38 was a decode
// stand-in without the RoPE step).
//
// Grid: one block per (batch, kv_head, position). Block: head_dim/2
// threads — each thread rotates one adjacent (even, odd) pair of
// the head_dim slice at this (b, h, p) triple.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void kv_cache_update_rope_bf16_kernel(
    const __nv_bfloat16 *__restrict__ k_in,
    const __nv_bfloat16 *__restrict__ cos_tbl,
    const __nv_bfloat16 *__restrict__ sin_tbl,
    __nv_bfloat16 *__restrict__ k_out,
    int32_t kv_heads,
    int32_t context,
    int32_t head_dim)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int p = blockIdx.z;
    const int pair = threadIdx.x;
    const int half = head_dim / 2;
    if (pair >= half) return;

    int base = (((b * kv_heads) + h) * context + p) * head_dim + 2 * pair;
    float r_even = __bfloat162float(k_in[base]);
    float r_odd  = __bfloat162float(k_in[base + 1]);
    float c = __bfloat162float(cos_tbl[p * half + pair]);
    float s = __bfloat162float(sin_tbl[p * half + pair]);
    k_out[base]     = __float2bfloat16(r_even * c - r_odd * s);
    k_out[base + 1] = __float2bfloat16(r_even * s + r_odd * c);
}

extern "C" void kv_cache_update_rope_bf16_launch(
    const __nv_bfloat16 *d_k_in,
    const __nv_bfloat16 *d_cos, const __nv_bfloat16 *d_sin,
    __nv_bfloat16 *d_k_out,
    int batch, int kv_heads, int context, int head_dim,
    cudaStream_t stream)
{
    dim3 grid(batch, kv_heads, context);
    dim3 block(head_dim / 2);
    kv_cache_update_rope_bf16_kernel<<<grid, block, 0, stream>>>(
        d_k_in, d_cos, d_sin, d_k_out, kv_heads, context, head_dim);
}
