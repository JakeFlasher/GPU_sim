// slug: hyena_fft_conv_block_fp32
// tier: P
// arch_gate: sm_90a sm_120
//
// Hyena FFT-conv block (shape-matched direct-conv reference). The full
// production Hyena block uses FFT for the long convolution; the reference
// family `ssm.py::_direct_conv` does a simple causal direct convolution
// (left-padded with zeros). That is the correctness contract: output at
// `out[ba][t][c]` is `sum_{k=0..kernel-1, tau=t-k >= 0} x[ba][tau][c] * filt[c][k]`.
// FP32 throughout; tolerance band is the (FP32, fma_chain, ssm) triplet.
//
// Grid: one block per (batch, channels-tile). Each thread owns one (t, c)
// output cell; the inner loop walks the kernel taps. Simple and portable
// across sm_90a and sm_120.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void hyena_fft_conv_block_fp32_kernel(
    const float *__restrict__ x,        // [batch, seq, channels]
    const float *__restrict__ filt,     // [channels, kernel]
    float *__restrict__ out,             // [batch, seq, channels]
    int32_t seq,
    int32_t channels,
    int32_t kernel_size)
{
    const int ba = blockIdx.x;
    const int t  = blockIdx.y * blockDim.y + threadIdx.y;
    const int c  = blockIdx.z * blockDim.x + threadIdx.x;
    if (t >= seq || c >= channels) return;

    float acc = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int tau = t - k;
        if (tau >= 0 && tau < seq) {
            acc += x[((ba * seq) + tau) * channels + c] * filt[c * kernel_size + k];
        }
    }
    out[((ba * seq) + t) * channels + c] = acc;
}

extern "C" void hyena_fft_conv_block_fp32_launch(
    const float *d_x,
    const float *d_filt,
    float *d_out,
    int batch, int seq, int channels, int kernel_size, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid(batch, (seq + block.y - 1) / block.y, (channels + block.x - 1) / block.x);
    hyena_fft_conv_block_fp32_kernel<<<grid, block, 0, stream>>>(
        d_x, d_filt, d_out, seq, channels, kernel_size);
}
