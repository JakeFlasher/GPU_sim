#include "kernel.h"
#include <cuda_runtime.h>

// Depthwise 1D conv (kernel_size=3, padding=2 then truncate to
// seq_len) followed by channel-split and gated multiplication.
//
// Per-output math (zero-padding boundary handled inline):
//   y[b, ch, t] = bias[ch]
//               + w[ch, 0] * u[b, ch, t - 2]   if t - 2 >= 0
//               + w[ch, 1] * u[b, ch, t - 1]   if t - 1 >= 0
//               + w[ch, 2] * u[b, ch, t]       (always in range)
//
// Channel split (inner_width = d_model * 3 = 768):
//   ch in [0,   256) -> x0[b, ch,         t]
//   ch in [256, 512) -> x1[b, ch - 256,   t]
//   ch in [512, 768) -> v;  v_gated[b, ch - 512, t] = v * x0[b, ch - 512, t]
//
// Phase 1 kernel writes x0 + x1; phase 2 kernel writes v_gated
// (recomputing the v-channel conv and gating against x0). Two
// kernels avoids needing an intermediate v_temp buffer the
// eval harness does not pre-allocate.

namespace {

constexpr int D_MODEL = 256;            // pinned by definition.json
constexpr int INNER_WIDTH = 768;        // d_model * (hyena_order + 1)
constexpr int SHORT_FILTER_ORDER = 3;   // pinned by definition.json
constexpr int BLOCK_THREADS = 256;

// Phase 1: emit conv outputs for the first 512 channels into the
// x0 (channels 0..255) and x1 (channels 256..511) destination
// tensors. Grid: (batch, 512, ceil(seq_len / BLOCK_THREADS)). Each
// thread handles one (b, ch, t) output.
__global__ void __launch_bounds__(BLOCK_THREADS, 4)
problem_006_hyena_depthwise_conv1d_split_gate_kernel(
    const float* __restrict__ u,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ x0,
    float* __restrict__ x1,
    int batch_size,
    int seq_len) {

    const int b = blockIdx.x;
    const int ch = blockIdx.y;                              // 0..511
    const int t = blockIdx.z * BLOCK_THREADS + threadIdx.x;
    if (b >= batch_size || ch >= 2 * D_MODEL || t >= seq_len) return;

    const float* u_row = u + (b * INNER_WIDTH + ch) * seq_len;
    const float* w_row = weight + ch * SHORT_FILTER_ORDER;
    const float w0 = __ldg(w_row + 0);
    const float w1 = __ldg(w_row + 1);
    const float w2 = __ldg(w_row + 2);
    const float bv = __ldg(bias + ch);

    float acc = bv;
    if (t >= 2) acc += w0 * __ldg(u_row + (t - 2));
    if (t >= 1) acc += w1 * __ldg(u_row + (t - 1));
    acc           += w2 * __ldg(u_row + t);

    if (ch < D_MODEL) {
        // x0[b, ch, t]
        x0[(b * D_MODEL + ch) * seq_len + t] = acc;
    } else {
        // x1[b, ch - D_MODEL, t]
        const int x1_ch = ch - D_MODEL;
        x1[(b * D_MODEL + x1_ch) * seq_len + t] = acc;
    }
}

// Phase 2: recompute conv for the v channels (ch in [512, 768))
// and gate by the already-written x0. Grid: (batch, D_MODEL,
// ceil(seq_len / BLOCK_THREADS)). Phase 2 depends on phase 1
// having completed (sol-execbench correctness path runs them
// serially on the same default stream).
__global__ void __launch_bounds__(BLOCK_THREADS, 4)
problem_006_hyena_depthwise_conv1d_split_gate_v_gate_kernel(
    const float* __restrict__ u,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ x0,
    float* __restrict__ v_gated,
    int batch_size,
    int seq_len) {

    const int b = blockIdx.x;
    const int ch_local = blockIdx.y;                        // 0..255 (v channel index)
    const int t = blockIdx.z * BLOCK_THREADS + threadIdx.x;
    if (b >= batch_size || ch_local >= D_MODEL || t >= seq_len) return;

    const int src_ch = ch_local + 2 * D_MODEL;              // 512..767 in u
    const float* u_row = u + (b * INNER_WIDTH + src_ch) * seq_len;
    const float* w_row = weight + src_ch * SHORT_FILTER_ORDER;
    const float w0 = __ldg(w_row + 0);
    const float w1 = __ldg(w_row + 1);
    const float w2 = __ldg(w_row + 2);
    const float bv = __ldg(bias + src_ch);

    float v_acc = bv;
    if (t >= 2) v_acc += w0 * __ldg(u_row + (t - 2));
    if (t >= 1) v_acc += w1 * __ldg(u_row + (t - 1));
    v_acc           += w2 * __ldg(u_row + t);

    const float x0_value = __ldg(x0 + (b * D_MODEL + ch_local) * seq_len + t);
    v_gated[(b * D_MODEL + ch_local) * seq_len + t] = v_acc * x0_value;
}

}  // namespace

void problem_006_hyena_depthwise_conv1d_split_gate_launcher(
    const torch::Tensor& u,
    const torch::Tensor& short_filter_weight,
    const torch::Tensor& short_filter_bias,
    torch::Tensor& v_gated,
    torch::Tensor& x0,
    torch::Tensor& x1,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(u.size(0));
    const int seq_len = static_cast<int>(u.size(2));
    const int t_blocks = (seq_len + BLOCK_THREADS - 1) / BLOCK_THREADS;

    dim3 grid_phase1(batch_size, 2 * D_MODEL, t_blocks);
    dim3 block(BLOCK_THREADS);
    problem_006_hyena_depthwise_conv1d_split_gate_kernel<<<grid_phase1, block, 0, stream>>>(
        u.data_ptr<float>(),
        short_filter_weight.data_ptr<float>(),
        short_filter_bias.data_ptr<float>(),
        x0.data_ptr<float>(),
        x1.data_ptr<float>(),
        batch_size,
        seq_len);

    dim3 grid_phase2(batch_size, D_MODEL, t_blocks);
    problem_006_hyena_depthwise_conv1d_split_gate_v_gate_kernel<<<grid_phase2, block, 0, stream>>>(
        u.data_ptr<float>(),
        short_filter_weight.data_ptr<float>(),
        short_filter_bias.data_ptr<float>(),
        x0.data_ptr<float>(),
        v_gated.data_ptr<float>(),
        batch_size,
        seq_len);
}
