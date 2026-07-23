#pragma once

#ifndef PROBLEM_068_FUSED_ROPE_BACKWARD_KERNEL_H
#define PROBLEM_068_FUSED_ROPE_BACKWARD_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher set for the fp32 fused RoPE-application backward
// kernel. The reference forward is:
//   output[..., d] = x[..., d] * cos[..., d] +
//                    rotate_half(x)[..., d] * sin[..., d]
// where rotate_half(x)[d] = -x[d + 64] for d in [0, 64)
//                        =  x[d - 64] for d in [64, 128)
//
// Backward gradients (per the definition):
//   grad_x   = grad_output * cos - rotate_half(grad_output) * sin
//   grad_cos = sum_h(grad_q_embed * q) + sum_h_kv(grad_k_embed * k)
//   grad_sin = sum_h(grad_q_embed * rotate_half(q)) +
//              sum_h_kv(grad_k_embed * rotate_half(k))
//
// head_dim = 128 and half_head_dim = 64 are pinned. The DPS entry
// launches two kernels in sequence:
//   * `problem_068_grad_x_launcher` is invoked TWICE — once for
//     (grad_q_embed, grad_q) using num_heads, once for
//     (grad_k_embed, grad_k) using num_kv_heads.
//   * `problem_068_grad_cos_sin_launcher` is invoked ONCE; it
//     accumulates the per-(b, s, d) reduction over BOTH H and
//     H_kv axes in a single CTA.
void problem_068_grad_x_launcher(
    const torch::Tensor& grad_x_embed,
    const torch::Tensor& cos_table,
    const torch::Tensor& sin_table,
    torch::Tensor& grad_x,
    cudaStream_t stream);

void problem_068_grad_cos_sin_launcher(
    const torch::Tensor& grad_q_embed,
    const torch::Tensor& grad_k_embed,
    const torch::Tensor& q,
    const torch::Tensor& k,
    torch::Tensor& grad_cos,
    torch::Tensor& grad_sin,
    cudaStream_t stream);

#endif  // PROBLEM_068_FUSED_ROPE_BACKWARD_KERNEL_H
