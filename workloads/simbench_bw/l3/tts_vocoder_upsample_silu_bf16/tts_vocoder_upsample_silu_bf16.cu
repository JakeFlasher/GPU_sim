// slug: tts_vocoder_upsample_silu_bf16
// tier: P
// arch_gate: sm_90a sm_120
//
// TTS vocoder: learned per-channel 1D transposed-conv upsample with
// stride=UF, kernel=UF, followed by SiLU.
// Matches references/tts.py::_vocoder_upsample_silu (plan-row 68
// contract: "transposed / upsample conv + SiLU").
//
// Y[b, c, l*UF + k] = silu(W_upsample[c, k] * X[b, c, l])
//
// Grid (B, C); each block scans Lout = L*UF output positions. Output
// index i splits into (input_pos = i / UF, within-group k = i % UF).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ void tts_vocoder_upsample_silu_bf16_kernel(
    const __nv_bfloat16 *__restrict__ X,
    const __nv_bfloat16 *__restrict__ W_upsample,    // [C, UF]
    __nv_bfloat16 *__restrict__ out,
    int32_t C, int32_t L, int32_t UF)
{
    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    const int Lout = L * UF;
    const size_t in_base  = ((size_t)b * C + c) * L;
    const size_t out_base = ((size_t)b * C + c) * Lout;
    const size_t w_base   = (size_t)c * UF;
    for (int i = tid; i < Lout; i += lanes) {
        int input_pos = i / UF;
        int k         = i - input_pos * UF;
        float xv = __bfloat162float(X[in_base + input_pos]);
        float wv = __bfloat162float(W_upsample[w_base + k]);
        float prod = wv * xv;
        float activated = prod / (1.0f + __expf(-prod));
        out[out_base + i] = __float2bfloat16(activated);
    }
}

extern "C" void tts_vocoder_upsample_silu_bf16_launch(
    const __nv_bfloat16 *d_X,
    const __nv_bfloat16 *d_W_upsample,
    __nv_bfloat16 *d_out,
    int B, int C, int L, int UF,
    cudaStream_t stream)
{
    int threads = 256;
    dim3 grid(B, C);
    dim3 block(threads);
    tts_vocoder_upsample_silu_bf16_kernel<<<grid, block, 0, stream>>>(
        d_X, d_W_upsample, d_out, C, L, UF);
}
