#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Host launcher for the Hyena depthwise 1D conv + split + gate
// kernel set. Two device kernels: phase 1 emits the x0 / x1
// streams (channels 0..511) directly into the destination
// tensors; phase 2 recomputes the v-channel conv (channels
// 512..767) and gates it against the already-written x0 stream
// to produce v_gated. The recompute is cheap (3 multiply-adds
// per output element) and avoids an intermediate v_temp buffer
// the eval harness does not pre-allocate.
//
// All inputs / outputs are FP32 per the upstream
// definition.json contract. Destination-passing style: x0,
// x1, and v_gated are pre-allocated by the eval driver and
// written in-place.
void problem_006_hyena_depthwise_conv1d_split_gate_launcher(
    const torch::Tensor& u,
    const torch::Tensor& short_filter_weight,
    const torch::Tensor& short_filter_bias,
    torch::Tensor& v_gated,
    torch::Tensor& x0,
    torch::Tensor& x1,
    cudaStream_t stream);
