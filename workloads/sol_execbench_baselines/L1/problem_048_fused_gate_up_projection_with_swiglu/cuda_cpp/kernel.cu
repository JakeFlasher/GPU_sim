#include "kernel.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

// Fused per-element GELU-tanh + bf16 multiply kernel.
//
// Reads bf16 gate_output and up_output; for each element:
//   - upcasts gate_output to fp32
//   - computes GELU-tanh in fp32:
//       arg      = sqrt(2/pi) * (g + 0.044715 * g^3)
//       act_fp32 = 0.5 * g * (1 + tanh(arg))
//   - downcasts activated gate to bf16
//   - multiplies by bf16 up_output, writes bf16 output
//
// This matches the reference's exact dtype path
// (`activated_gate.to(gate_output.dtype) * up_output`), which is
// load-bearing for the workload tolerance band: doing the multiply
// in fp32 and casting at the end produces a value subtly different
// from the bf16-cast-then-bf16-multiply path the reference uses.

namespace {

constexpr int kBlockThreads = 256;

__global__ void __launch_bounds__(kBlockThreads, 4) k_fused_gelu_tanh_mul(
    const __nv_bfloat16* __restrict__ gate_output,
    const __nv_bfloat16* __restrict__ up_output,
    __nv_bfloat16* __restrict__ output,
    size_t total_elements
) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float c = 0.044715f;

    const size_t idx_base =
        (size_t)blockIdx.x * kBlockThreads + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * kBlockThreads;
    for (size_t idx = idx_base; idx < total_elements; idx += stride) {
        const float g = __bfloat162float(gate_output[idx]);
        const float g2 = g * g;
        const float g3 = g2 * g;
        const float arg = sqrt_2_over_pi * (g + c * g3);
        // Keep the tanh tail accurate even when the baseline's
        // --use_fast_math compile option is present.  The reference's
        // tolerance reaches the saturated-GELU region, where tanh.approx
        // rounds 1+tanh(arg) to zero while the FP32 reference retains a
        // small nonzero value. Evaluate tanh in double, then round to FP32
        // before the original BF16 activation/product rounding steps.
        const float t = static_cast<float>(tanh(static_cast<double>(arg)));
        const float act_fp32 = 0.5f * g * (1.0f + t);
        const __nv_bfloat16 act_bf16 = __float2bfloat16(act_fp32);
        const __nv_bfloat16 up_bf16 = up_output[idx];
        output[idx] = __hmul(act_bf16, up_bf16);
    }
}

}  // namespace

void problem_048_fused_gelu_tanh_mul_launcher(
    const torch::Tensor& gate_output,
    const torch::Tensor& up_output,
    torch::Tensor& output,
    cudaStream_t stream
) {
    const size_t total = (size_t)gate_output.numel();
    const int max_blocks = 65535;
    int blocks = static_cast<int>(
        (total + kBlockThreads - 1) / kBlockThreads);
    if (blocks > max_blocks) blocks = max_blocks;
    dim3 grid(blocks);
    dim3 block(kBlockThreads);
    k_fused_gelu_tanh_mul<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            gate_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            up_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            output.data_ptr<at::BFloat16>()),
        total);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "problem_048 fused GELU-tanh + multiply kernel "
                "launch failed");
}
