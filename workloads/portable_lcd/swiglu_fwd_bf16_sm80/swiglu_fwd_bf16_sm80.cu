// slug: swiglu_fwd_bf16_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// BF16 SwiGLU activation, element-wise:
//
//   out[i] = SiLU(gate[i]) * up[i]
//          = (gate[i] / (1 + exp(-gate[i]))) * up[i]
//
// Llama-style FFN gate path. No reduction, no shared memory, no
// MMA — pure element-wise stream over the input pair. Each thread
// handles one or more strided indices; no inter-thread
// communication is needed.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void swiglu_fwd_bf16_sm80_kernel(
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    __nv_bfloat16       *__restrict__ out,
    int64_t n)
{
    const int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x  * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float silu = g / (1.0f + expf(-g));
        out[i] = __float2bfloat16(silu * u);
    }
}

extern "C" void swiglu_fwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_gate,
    const __nv_bfloat16 *d_up,
    __nv_bfloat16 *d_out,
    int64_t n,
    cudaStream_t stream)
{
    const int block = 256;
    int grid = (int)((n + block - 1) / block);
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;
    swiglu_fwd_bf16_sm80_kernel<<<grid, block, 0, stream>>>(d_gate, d_up, d_out, n);
}
