// slug: geglu_ffn_fused_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// Fused GeGLU FFN: out[t,h] = sum_c gelu(gate[t,c]) * up[t,c] * w_down[c,h]
// where gate = x @ w_gate and up = x @ w_up. Identical layout to SwiGLU
// but with GeLU (tanh approximation) on the gated branch, matching
// references/activation.py for slug "geglu_ffn_fused_bf16".

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Block layout: block.x covers the output's hidden dim in chunks of 256.
extern "C" __global__ void geglu_ffn_fused_bf16_kernel(
    const __nv_bfloat16 *__restrict__ x,       // [tokens, hidden]
    const __nv_bfloat16 *__restrict__ w_gate,  // [hidden, intermediate]
    const __nv_bfloat16 *__restrict__ w_up,    // [hidden, intermediate]
    const __nv_bfloat16 *__restrict__ w_down,  // [intermediate, hidden]
    __nv_bfloat16 *__restrict__ out,           // [tokens, hidden]
    int32_t hidden,
    int32_t intermediate)
{
    const int token = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;

    float acc = 0.0f;
    for (int c = 0; c < intermediate; ++c) {
        // gate[t, c] = dot(x[t, :], w_gate[:, c])
        float gate_c = 0.0f;
        float up_c = 0.0f;
        for (int k = 0; k < hidden; ++k) {
            float xv = __bfloat162float(x[token * hidden + k]);
            gate_c += xv * __bfloat162float(w_gate[k * intermediate + c]);
            up_c  += xv * __bfloat162float(w_up  [k * intermediate + c]);
        }
        // GeLU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
        float inner = 0.7978845608f * (gate_c + 0.044715f * gate_c * gate_c * gate_c);
        float gelu = 0.5f * gate_c * (1.0f + tanhf(inner));
        acc += gelu * up_c * __bfloat162float(w_down[c * hidden + h]);
    }
    out[token * hidden + h] = __float2bfloat16(acc);
}

extern "C" void geglu_ffn_fused_bf16_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_w_gate,
    const __nv_bfloat16 *d_w_up,
    const __nv_bfloat16 *d_w_down,
    __nv_bfloat16 *d_out,
    int tokens, int hidden, int intermediate, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((hidden + block.x - 1) / block.x, tokens);
    geglu_ffn_fused_bf16_kernel<<<grid, block, 0, stream>>>(
        d_x, d_w_gate, d_w_up, d_w_down, d_out, hidden, intermediate);
}
