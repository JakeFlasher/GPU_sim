#include "kernel.h"
#include <cuda_runtime.h>

// fp32 multi-axis RoPE frequency computation for FLUX 3D position
// encoding (time, height, width). Each output element is a
// pure-compute sin/cos of an angle whose construction depends on
// a single ids[s, axis] load plus an on-the-fly inverse-frequency
// table entry computed via `powf(theta, k / half_dim_axis)`.
//
// Grid: (seq_len,), one CTA per seq_len position. Each CTA has
// BLOCK_THREADS = total_dim = 128 (one thread per output d).
//
// Per-axis pinned constants (FLUX-Fill-dev):
//   axes_dim = {16, 56, 56},  total_dim = 128
//   axis_offset = {0, 16, 72}, half_dim = {8, 28, 28}
//
// Read traffic per CTA: 3 fp32 loads of ids[s, 0..2] (one broadcast
// per axis range across 16 / 56 / 56 threads respectively) plus
// the immediate `theta` constant. Output traffic dominates:
// 128 fp32 cos writes + 128 fp32 sin writes per CTA.

namespace {

constexpr int kNumAxes = 3;
constexpr int kTotalDim = 128;
constexpr int kBlockThreads = kTotalDim;

// Each thread maps its `d` in [0, 128) to (axis, pair_idx, half_dim_f).
// Compile-time constant axis spec (matches the FLUX-Fill-dev pin).
struct AxisSpec {
    int axis;
    int pair_idx;
    float half_dim_f;
};

__device__ inline AxisSpec resolve_axis(int d) {
    AxisSpec s;
    if (d < 16) {
        // axis 0: d in [0, 16), half_dim = 8
        s.axis = 0;
        s.pair_idx = d >> 1;
        s.half_dim_f = 8.0f;
    } else if (d < 72) {
        // axis 1: d in [16, 72), half_dim = 28
        s.axis = 1;
        s.pair_idx = (d - 16) >> 1;
        s.half_dim_f = 28.0f;
    } else {
        // axis 2: d in [72, 128), half_dim = 28
        s.axis = 2;
        s.pair_idx = (d - 72) >> 1;
        s.half_dim_f = 28.0f;
    }
    return s;
}

__global__ void __launch_bounds__(kBlockThreads, 8)
problem_034_rope_freq_kernel(
    const float* __restrict__ ids,
    float theta,
    float* __restrict__ freqs_cos,
    float* __restrict__ freqs_sin,
    int seq_len) {

    const int s = blockIdx.x;
    const int d = threadIdx.x;
    if (s >= seq_len) return;

    const AxisSpec spec = resolve_axis(d);

    // Read this row's axis position (broadcast across threads sharing
    // the same axis).
    const float axis_pos = ids[s * kNumAxes + spec.axis];

    // Compute freq_band on the fly:
    //   freq_band = 1.0 / theta^(pair_idx / half_dim)
    //             = exp(-(pair_idx / half_dim) * log(theta))
    // Using powf preserves the reference's numerics exactly.
    const float exponent = static_cast<float>(spec.pair_idx) / spec.half_dim_f;
    const float freq_band = 1.0f / powf(theta, exponent);

    const float angle = axis_pos * freq_band;
    const int out_offset = s * kTotalDim + d;
    freqs_cos[out_offset] = cosf(angle);
    freqs_sin[out_offset] = sinf(angle);
}

}  // namespace

void problem_034_flux_multi_axis_rope_freq_launcher(
    const torch::Tensor& ids,
    float theta,
    torch::Tensor& freqs_cos,
    torch::Tensor& freqs_sin,
    cudaStream_t stream) {

    const int seq_len = static_cast<int>(ids.size(0));

    dim3 grid(seq_len);
    dim3 block(kBlockThreads);

    problem_034_rope_freq_kernel<<<grid, block, 0, stream>>>(
        ids.data_ptr<float>(),
        theta,
        freqs_cos.data_ptr<float>(),
        freqs_sin.data_ptr<float>(),
        seq_len);
}
