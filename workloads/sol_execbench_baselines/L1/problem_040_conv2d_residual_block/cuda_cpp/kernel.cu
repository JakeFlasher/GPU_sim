#include "kernel.h"
#include <cuda_runtime.h>

// Conv2d residual block (3 -> 32 -> 3 channels, kernel_size=3,
// padding=1) followed by residual addition: out = conv_out(
// conv_in(x)) + x. Two-kernel design: phase 1 produces the
// (B, HIDDEN_C=32, H, W) hidden tensor; phase 2 consumes
// hidden + x and writes the final output.
//
// Per-output math (zero-padding boundary handled inline):
//   y[b, oc, h, w] = bias[oc]
//                  + sum_{ic, ki, kj} w[oc, ic, ki, kj]
//                                    * x[b, ic, h + ki - 1, w + kj - 1]
//                                    when the sampled index lies inside
//                                    the H x W frame; otherwise 0.

namespace {

constexpr int IN_CHANNELS = 3;       // pinned by definition.json
constexpr int HIDDEN_CHANNELS = 32;  // pinned by definition.json
constexpr int OUT_CHANNELS = 3;      // pinned by definition.json
constexpr int KERNEL_SIZE = 3;       // pinned by definition.json
constexpr int PAD = 1;               // (kernel_size - 1) / 2
constexpr int BLOCK_W = 16;
constexpr int BLOCK_H = 16;

// Phase 1: conv_in (3 -> 32 channels). Grid: (ceil(W/BLOCK_W),
// ceil(H/BLOCK_H), batch * HIDDEN_CHANNELS). Block: BLOCK_W x BLOCK_H.
__global__ void __launch_bounds__(BLOCK_W * BLOCK_H, 4)
problem_040_conv2d_residual_block_conv_in_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_in,
    const float* __restrict__ b_in,
    float* __restrict__ hidden,
    int batch_size,
    int height,
    int width) {

    const int w = blockIdx.x * BLOCK_W + threadIdx.x;
    const int h = blockIdx.y * BLOCK_H + threadIdx.y;
    const int bo = blockIdx.z;
    const int b = bo / HIDDEN_CHANNELS;
    const int oc = bo % HIDDEN_CHANNELS;
    if (w >= width || h >= height || b >= batch_size) return;

    const int spatial = height * width;
    const int batch_stride = IN_CHANNELS * spatial;
    const float* x_b = x + b * batch_stride;

    float acc = __ldg(b_in + oc);
    #pragma unroll
    for (int ic = 0; ic < IN_CHANNELS; ++ic) {
        const float* x_ic = x_b + ic * spatial;
        #pragma unroll
        for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
            const int ih = h + ki - PAD;
            if (ih < 0 || ih >= height) continue;
            #pragma unroll
            for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
                const int iw = w + kj - PAD;
                if (iw < 0 || iw >= width) continue;
                const float wv = __ldg(
                    w_in
                    + ((oc * IN_CHANNELS + ic) * KERNEL_SIZE + ki)
                        * KERNEL_SIZE + kj);
                const float xv = __ldg(x_ic + ih * width + iw);
                acc += wv * xv;
            }
        }
    }
    hidden[((b * HIDDEN_CHANNELS + oc) * height + h) * width + w] = acc;
}

// Phase 2: conv_out (32 -> 3 channels) + residual add. Grid same
// shape as phase 1 but channels = OUT_CHANNELS.
__global__ void __launch_bounds__(BLOCK_W * BLOCK_H, 4)
problem_040_conv2d_residual_block_conv_out_plus_residual_kernel(
    const float* __restrict__ hidden,
    const float* __restrict__ x,
    const float* __restrict__ w_out,
    const float* __restrict__ b_out,
    float* __restrict__ output,
    int batch_size,
    int height,
    int width) {

    const int w = blockIdx.x * BLOCK_W + threadIdx.x;
    const int h = blockIdx.y * BLOCK_H + threadIdx.y;
    const int bo = blockIdx.z;
    const int b = bo / OUT_CHANNELS;
    const int oc = bo % OUT_CHANNELS;
    if (w >= width || h >= height || b >= batch_size) return;

    const int spatial = height * width;
    const int hidden_batch_stride = HIDDEN_CHANNELS * spatial;
    const float* hidden_b = hidden + b * hidden_batch_stride;
    const float* x_b = x + b * (IN_CHANNELS * spatial);

    float acc = __ldg(b_out + oc);
    #pragma unroll
    for (int ic = 0; ic < HIDDEN_CHANNELS; ++ic) {
        const float* h_ic = hidden_b + ic * spatial;
        #pragma unroll
        for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
            const int ih = h + ki - PAD;
            if (ih < 0 || ih >= height) continue;
            #pragma unroll
            for (int kj = 0; kj < KERNEL_SIZE; ++kj) {
                const int iw = w + kj - PAD;
                if (iw < 0 || iw >= width) continue;
                const float wv = __ldg(
                    w_out
                    + ((oc * HIDDEN_CHANNELS + ic) * KERNEL_SIZE + ki)
                        * KERNEL_SIZE + kj);
                const float hv = __ldg(h_ic + ih * width + iw);
                acc += wv * hv;
            }
        }
    }
    // Residual add: output = conv_out(hidden) + x. The input
    // channel-count (IN_CHANNELS=3) matches OUT_CHANNELS, so the
    // residual lookup is along the same `oc` slot.
    const float xv = __ldg(x_b + (oc * height + h) * width + w);
    output[((b * OUT_CHANNELS + oc) * height + h) * width + w] = acc + xv;
}

}  // namespace

void problem_040_conv2d_residual_block_launcher(
    const torch::Tensor& x,
    const torch::Tensor& conv_in_weight,
    const torch::Tensor& conv_in_bias,
    const torch::Tensor& conv_out_weight,
    const torch::Tensor& conv_out_bias,
    torch::Tensor& output,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(x.size(0));
    const int height = static_cast<int>(x.size(2));
    const int width = static_cast<int>(x.size(3));

    // Allocate the (B, HIDDEN_CHANNELS, H, W) hidden tensor for
    // the conv_in -> conv_out chain. The launcher owns this buffer;
    // the eval harness does not pre-allocate it.
    auto hidden = torch::empty(
        {batch_size, HIDDEN_CHANNELS, height, width}, x.options());

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid_in(
        (width + BLOCK_W - 1) / BLOCK_W,
        (height + BLOCK_H - 1) / BLOCK_H,
        batch_size * HIDDEN_CHANNELS);
    problem_040_conv2d_residual_block_conv_in_kernel<<<grid_in, block, 0, stream>>>(
        x.data_ptr<float>(),
        conv_in_weight.data_ptr<float>(),
        conv_in_bias.data_ptr<float>(),
        hidden.data_ptr<float>(),
        batch_size,
        height,
        width);

    dim3 grid_out(
        (width + BLOCK_W - 1) / BLOCK_W,
        (height + BLOCK_H - 1) / BLOCK_H,
        batch_size * OUT_CHANNELS);
    problem_040_conv2d_residual_block_conv_out_plus_residual_kernel<<<grid_out, block, 0, stream>>>(
        hidden.data_ptr<float>(),
        x.data_ptr<float>(),
        conv_out_weight.data_ptr<float>(),
        conv_out_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        height,
        width);
}
