// slug: qkv_rope_yarn_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// YaRN-scaled RoPE applied to the fused QKV tensor. The rotation kernel
// itself is identical to the standard variant — the YaRN scaling is
// encoded into the host-precomputed cos/sin tables (each angle divided
// by yarn_scale before cos/sin). Matches references/rope.py for slug
// == "qkv_rope_yarn_bf16".

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void qkv_rope_yarn_bf16_kernel(
    __nv_bfloat16 *__restrict__ qkv,
    const __nv_bfloat16 *__restrict__ cos,
    const __nv_bfloat16 *__restrict__ sin,
    int32_t heads,
    int32_t head_dim)
{
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int pair = threadIdx.x;
    const int half = head_dim / 2;
    if (pair >= half) return;
    int base = ((token * heads) + head) * head_dim + 2 * pair;
    float r_even = __bfloat162float(qkv[base]);
    float r_odd  = __bfloat162float(qkv[base + 1]);
    float c = __bfloat162float(cos[token * half + pair]);
    float s = __bfloat162float(sin[token * half + pair]);
    qkv[base]     = __float2bfloat16(r_even * c - r_odd * s);
    qkv[base + 1] = __float2bfloat16(r_even * s + r_odd * c);
}

extern "C" void qkv_rope_yarn_bf16_launch(
    __nv_bfloat16 *d_qkv,
    const __nv_bfloat16 *d_cos,
    const __nv_bfloat16 *d_sin,
    int tokens, int heads, int head_dim, cudaStream_t stream)
{
    dim3 grid(tokens, heads);
    dim3 block(head_dim / 2);
    qkv_rope_yarn_bf16_kernel<<<grid, block, 0, stream>>>(d_qkv, d_cos, d_sin, heads, head_dim);
}
