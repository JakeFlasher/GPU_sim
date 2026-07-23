#include "kernel.h"
#include <cuda_runtime.h>

// FP32 GELU with tanh approximation:
//   output = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Streams x and writes output via float4 vectorized loads/stores
// when the element count is a multiple of 4 and the tensor base
// pointer is 16-byte aligned (always true for ATen-allocated CUDA
// tensors). For the tail, a scalar fallback handles any remainder.
//
// Numerical contract: the reference uses ``torch.tanh`` (FP32
// math.tanh-style). ``tanhf`` matches the reference's tolerance
// (max_atol=1e-5, max_rtol=0.05 per the workload set).

namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kCoeff = 0.044715f;

__device__ __forceinline__ float gelu_tanh_approx_f(float x) {
    float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void problem_025_gelu_vec4_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ output,
    int n_vec4) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec4) {
        return;
    }
    float4 in = x[idx];
    float4 out;
    out.x = gelu_tanh_approx_f(in.x);
    out.y = gelu_tanh_approx_f(in.y);
    out.z = gelu_tanh_approx_f(in.z);
    out.w = gelu_tanh_approx_f(in.w);
    output[idx] = out;
}

__global__ void problem_025_gelu_tail_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int tail_start,
    int total) {

    const int idx = tail_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    output[idx] = gelu_tanh_approx_f(x[idx]);
}

}  // namespace

void problem_025_video_latent_gelu_launcher(
    const torch::Tensor& x,
    torch::Tensor& output,
    cudaStream_t stream) {

    const int total = static_cast<int>(x.numel());
    const int n_vec4 = total / 4;
    const int tail_start = n_vec4 * 4;

    constexpr int BLOCK_THREADS = 256;

    if (n_vec4 > 0) {
        const int grid = (n_vec4 + BLOCK_THREADS - 1) / BLOCK_THREADS;
        problem_025_gelu_vec4_kernel<<<grid, BLOCK_THREADS, 0, stream>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            n_vec4);
    }
    if (tail_start < total) {
        const int remaining = total - tail_start;
        const int grid = (remaining + BLOCK_THREADS - 1) / BLOCK_THREADS;
        problem_025_gelu_tail_kernel<<<grid, BLOCK_THREADS, 0, stream>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            tail_start,
            total);
    }
}
