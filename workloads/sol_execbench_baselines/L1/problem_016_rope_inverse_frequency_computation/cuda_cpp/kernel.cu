#include "kernel.h"
#include <cuda_runtime.h>

// One CTA, half_head_dim threads. Each thread computes
// `inv_freq[i] = 1.0f / powf(rope_theta, (2*i) / head_dim)`. The
// problem definition pins head_dim = 128, half_head_dim = 64.

namespace {

constexpr int HEAD_DIM = 128;
constexpr int HALF_HEAD_DIM = 64;

__global__ void __launch_bounds__(HALF_HEAD_DIM, 16)
problem_016_rope_inverse_frequency_computation_kernel(
    float rope_theta,
    float* __restrict__ inv_freq) {

    const int i = threadIdx.x;
    if (i >= HALF_HEAD_DIM) return;
    // exponent = (2 * i) / head_dim
    const float exponent =
        static_cast<float>(2 * i) / static_cast<float>(HEAD_DIM);
    // inv_freq = 1 / rope_theta^exponent
    inv_freq[i] = 1.0f / powf(rope_theta, exponent);
}

}  // namespace

void problem_016_rope_inverse_frequency_computation_launcher(
    float rope_theta,
    torch::Tensor& inv_freq,
    cudaStream_t stream) {

    dim3 grid(1);
    dim3 block(HALF_HEAD_DIM);
    problem_016_rope_inverse_frequency_computation_kernel<<<
        grid, block, 0, stream>>>(
        rope_theta,
        inv_freq.data_ptr<float>());
}
