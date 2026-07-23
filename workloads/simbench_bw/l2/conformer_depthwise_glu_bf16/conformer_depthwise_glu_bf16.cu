// slug: conformer_depthwise_glu_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// Conformer convolution module: pointwise(C->2C) + GLU + depthwise
// conv1d over time (kernel_size) + SiLU + pointwise(C->C). Matches
// references/audio.py::_conformer_depthwise_glu.
//
// Two kernels chained through a shared [B, T, C] GLU buffer:
// (1) pointwise_glu_kernel: glu[b, t, c] = (x[b,t,:] @ W_in[:, c]) *
//     sigmoid(x[b,t,:] @ W_in[:, C+c]).
// (2) conv_silu_pointwise_kernel: out[b, t, c_out] = sum_{c_in} silu(
//     sum_{k} glu[b, t+k-pad, c_in] * W_dw[c_in, k] ) * W_out[c_in, c_out].

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>


extern "C" __global__ void conformer_pointwise_glu_kernel(
    const __nv_bfloat16 *__restrict__ x,      // [B, T, C]
    const __nv_bfloat16 *__restrict__ W_in,   // [C, 2*C]
    __nv_bfloat16 *__restrict__ glu,          // [B, T, C]
    int32_t B, int32_t T, int32_t C)
{
    const int b = blockIdx.x;
    const int t = blockIdx.y;
    if (b >= B || t >= T) return;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    for (int c = tid; c < C; c += lanes) {
        float input_p = 0.0f, gate_p = 0.0f;
        for (int ci = 0; ci < C; ++ci) {
            float xv = __bfloat162float(x[(b * T + t) * C + ci]);
            input_p += xv * __bfloat162float(W_in[ci * (2 * C) + c]);
            gate_p  += xv * __bfloat162float(W_in[ci * (2 * C) + C + c]);
        }
        float sig = 1.0f / (1.0f + __expf(-gate_p));
        glu[(b * T + t) * C + c] = __float2bfloat16(input_p * sig);
    }
}


extern "C" __global__ void conformer_conv_silu_pointwise_kernel(
    const __nv_bfloat16 *__restrict__ glu,    // [B, T, C]
    const __nv_bfloat16 *__restrict__ W_dw,   // [C, Ks]
    const __nv_bfloat16 *__restrict__ W_out,  // [C, C]
    __nv_bfloat16 *__restrict__ out,          // [B, T, C]
    int32_t B, int32_t T, int32_t C, int32_t Ks)
{
    const int b = blockIdx.x;
    const int t = blockIdx.y;
    if (b >= B || t >= T) return;
    const int c_out = threadIdx.x;
    const int lanes = blockDim.x;
    const int pad = Ks / 2;

    for (int co = c_out; co < C; co += lanes) {
        float acc = 0.0f;
        for (int ci = 0; ci < C; ++ci) {
            // depthwise conv1d on channel ci around t
            float conv_v = 0.0f;
            for (int k = 0; k < Ks; ++k) {
                int src = t + k - pad;
                if (src < 0 || src >= T) continue;
                float gv = __bfloat162float(glu[(b * T + src) * C + ci]);
                float wv = __bfloat162float(W_dw[ci * Ks + k]);
                conv_v += gv * wv;
            }
            // SiLU
            float silu = conv_v / (1.0f + __expf(-conv_v));
            acc += silu * __bfloat162float(W_out[ci * C + co]);
        }
        out[(b * T + t) * C + co] = __float2bfloat16(acc);
    }
}


extern "C" void conformer_depthwise_glu_bf16_launch(
    const __nv_bfloat16 *d_x,
    const __nv_bfloat16 *d_W_in,
    const __nv_bfloat16 *d_W_dw,
    const __nv_bfloat16 *d_W_out,
    __nv_bfloat16 *d_glu_scratch,
    __nv_bfloat16 *d_out,
    int B, int T, int C, int Ks,
    cudaStream_t stream)
{
    dim3 grid(B, T);
    int threads = C;
    if (threads < 32) threads = 32;
    if (threads > 256) threads = 256;
    dim3 block(threads);
    conformer_pointwise_glu_kernel<<<grid, block, 0, stream>>>(
        d_x, d_W_in, d_glu_scratch, B, T, C);
    conformer_conv_silu_pointwise_kernel<<<grid, block, 0, stream>>>(
        d_glu_scratch, d_W_dw, d_W_out, d_out, B, T, C, Ks);
}
