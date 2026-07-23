#include "kernel.h"
#include <cuda_runtime.h>

// FP32 SiLU (Swish) backward pass:
//   grad_input = grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x))
//
// All four tensors are flat [num_elements] FP32 row-major. The
// reference caches `sigmoid_x` from the forward pass so the backward
// kernel does not need to recompute the sigmoid; the algebraic form
// above is identical to `d/dx[silu(x)] = sigmoid(x) * (1 + x * (1 -
// sigmoid(x)))` once the chain rule with the upstream `grad_output`
// is applied.
//
// Streams the three inputs and writes `grad_input` via float4
// vectorized loads/stores when the element count is a multiple of 4
// (always true for ATen-allocated CUDA tensors of fp32 with a power-
// of-2-aligned base; the tail kernel handles any remainder).

namespace {

__device__ __forceinline__ float silu_backward_elem(
    float go, float xv, float sx) {
    return go * sx * (1.0f + xv * (1.0f - sx));
}

__global__ void problem_084_silu_bwd_vec4_kernel(
    const float4* __restrict__ grad_output,
    const float4* __restrict__ x,
    const float4* __restrict__ sigmoid_x,
    float4* __restrict__ grad_input,
    int n_vec4) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec4) {
        return;
    }
    const float4 go = grad_output[idx];
    const float4 xv = x[idx];
    const float4 sx = sigmoid_x[idx];

    float4 out;
    out.x = silu_backward_elem(go.x, xv.x, sx.x);
    out.y = silu_backward_elem(go.y, xv.y, sx.y);
    out.z = silu_backward_elem(go.z, xv.z, sx.z);
    out.w = silu_backward_elem(go.w, xv.w, sx.w);
    grad_input[idx] = out;
}

__global__ void problem_084_silu_bwd_tail_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ x,
    const float* __restrict__ sigmoid_x,
    float* __restrict__ grad_input,
    int tail_start,
    int total) {

    const int idx = tail_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    grad_input[idx] = silu_backward_elem(
        grad_output[idx], x[idx], sigmoid_x[idx]);
}

}  // namespace

void problem_084_silu_activation_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& sigmoid_x,
    torch::Tensor& grad_input,
    cudaStream_t stream) {

    const int total = static_cast<int>(grad_output.numel());
    const int n_vec4 = total / 4;
    const int tail_start = n_vec4 * 4;

    constexpr int BLOCK_THREADS = 256;

    if (n_vec4 > 0) {
        const int grid = (n_vec4 + BLOCK_THREADS - 1) / BLOCK_THREADS;
        problem_084_silu_bwd_vec4_kernel<<<grid, BLOCK_THREADS, 0, stream>>>(
            reinterpret_cast<const float4*>(grad_output.data_ptr<float>()),
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<const float4*>(sigmoid_x.data_ptr<float>()),
            reinterpret_cast<float4*>(grad_input.data_ptr<float>()),
            n_vec4);
    }
    if (tail_start < total) {
        const int remaining = total - tail_start;
        const int grid = (remaining + BLOCK_THREADS - 1) / BLOCK_THREADS;
        problem_084_silu_bwd_tail_kernel<<<grid, BLOCK_THREADS, 0, stream>>>(
            grad_output.data_ptr<float>(),
            x.data_ptr<float>(),
            sigmoid_x.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            tail_start,
            total);
    }
}
