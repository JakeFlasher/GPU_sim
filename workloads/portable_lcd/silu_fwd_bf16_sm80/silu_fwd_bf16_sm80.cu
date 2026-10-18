// slug: silu_fwd_bf16_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// BF16 SiLU activation, element-wise:
//
//   out[i] = SiLU(x[i]) = x[i] / (1 + exp(-x[i]))
//
// Sister kernel of `swiglu_fwd_bf16_sm80` (which computes
// `SiLU(gate) * up`); this kernel is the unfused single-input
// SiLU and contributes the bare activation opcode mix to the
// suite (no second multiply, no input pair). No reduction, no
// shared memory, no MMA — pure element-wise stream over the
// input.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void silu_fwd_bf16_sm80_kernel(
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16       *__restrict__ out,
    int64_t n)
{
    const int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x  * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float v    = __bfloat162float(x[i]);
        float silu = v / (1.0f + expf(-v));
        out[i] = __float2bfloat16(silu);
    }
}

extern "C" void silu_fwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_x,
    __nv_bfloat16 *d_out,
    int64_t n,
    cudaStream_t stream)
{
    const int block = 256;
    int grid = (int)((n + block - 1) / block);
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;
    silu_fwd_bf16_sm80_kernel<<<grid, block, 0, stream>>>(d_x, d_out, n);
}
