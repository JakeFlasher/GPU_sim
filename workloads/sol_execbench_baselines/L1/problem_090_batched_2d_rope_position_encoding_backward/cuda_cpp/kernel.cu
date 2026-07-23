#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// BF16/fp32 batched 2D RoPE position-encoding backward.
//
// Per-element compute (no reductions, no broadcasts):
//   grad_idx_theta[b, s, d] = -grad_cos[b, s, d].f32 * sin(idx_theta[b, s, d])
//                            + grad_sin[b, s, d].f32 * cos(idx_theta[b, s, d])
//
// Each thread does:
//   * load fp32 idx_theta,
//   * SFU sinf + cosf,
//   * load BF16 grad_cos and grad_sin, cast to fp32,
//   * fused multiply-add to produce the fp32 output,
//   * store fp32 grad_idx_theta.
//
// One CTA per (b, s) row. Grid: (seq_len, batch_size).
// BLOCK_THREADS = 256; each thread stride-loops over head_dim.

namespace {

constexpr int kBlockThreads = 256;

__global__ void __launch_bounds__(kBlockThreads, 4)
problem_090_grad_idx_theta_kernel(
    const __nv_bfloat16* __restrict__ grad_cos,
    const __nv_bfloat16* __restrict__ grad_sin,
    const float* __restrict__ idx_theta,
    float* __restrict__ grad_idx_theta,
    int head_dim) {

    const int s = blockIdx.x;
    const int b = blockIdx.y;
    const int seq_len = gridDim.x;
    const int row_offset =
        (b * seq_len + s) * head_dim;

    const int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += kBlockThreads) {
        const int idx = row_offset + d;

        const float idx_t = idx_theta[idx];
        const float sin_t = sinf(idx_t);
        const float cos_t = cosf(idx_t);

        const float g_cos = __bfloat162float(grad_cos[idx]);
        const float g_sin = __bfloat162float(grad_sin[idx]);

        grad_idx_theta[idx] = -g_cos * sin_t + g_sin * cos_t;
    }
}

}  // namespace

void problem_090_batched_2d_rope_backward_launcher(
    const torch::Tensor& grad_cos,
    const torch::Tensor& grad_sin,
    const torch::Tensor& idx_theta,
    torch::Tensor& grad_idx_theta,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(idx_theta.size(0));
    const int seq_len = static_cast<int>(idx_theta.size(1));
    const int head_dim = static_cast<int>(idx_theta.size(2));

    dim3 grid(seq_len, batch_size);
    dim3 block(kBlockThreads);

    problem_090_grad_idx_theta_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            grad_cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            grad_sin.data_ptr<at::BFloat16>()),
        idx_theta.data_ptr<float>(),
        grad_idx_theta.data_ptr<float>(),
        head_dim);
}
