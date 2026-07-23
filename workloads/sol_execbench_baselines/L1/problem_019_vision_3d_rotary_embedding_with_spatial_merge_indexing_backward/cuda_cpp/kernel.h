#pragma once

#ifndef PROBLEM_019_VISION_3D_ROPE_BACKWARD_KERNEL_H
#define PROBLEM_019_VISION_3D_ROPE_BACKWARD_KERNEL_H

#include <cuda_runtime.h>
#include <torch/extension.h>

// Host launcher set for the fp32 Vision 3D RoPE backward kernels.
//
// Forward (from definition):
//   cos = cos(embeddings)               # [S, D]
//   sin = sin(embeddings)               # [S, D]
//   q_embed = q * cos + rotate_half(q) * sin
//   k_embed = k * cos + rotate_half(k) * sin
// where rotate_half(x)[d] = -x[d + half_dim] for d < half_dim
//                         =  x[d - half_dim] for d >= half_dim
//
// Backward:
//   grad_x[s, h, d] = grad_x_embed[s, h, d] * cos(emb[s, d])
//                   - partner_sign_fwd[d] * grad_x_embed[s, h, partner_d]
//                     * sin(emb[s, partner_d])
//   grad_embeddings[s, d] = -grad_cos_total[s, d] * sin(emb[s, d])
//                         +  grad_sin_total[s, d] * cos(emb[s, d])
//   where grad_cos_total[s, d] = Σ_h (g_q[s,h,d]*q[s,h,d] + g_k[s,h,d]*k[s,h,d])
//         grad_sin_total[s, d] = partner_sign_fwd[d] *
//                                Σ_h (g_q[s,h,d]*q[s,h,partner_d]
//                                    + g_k[s,h,d]*k[s,h,partner_d])
//   partner_sign_fwd[d] = -1 for d < half_dim, +1 for d >= half_dim
//
// head_dim = 128 and num_heads = 16 are pinned by the problem
// definition; half_head_dim = 64. seq_len is workload-variable.
//
// Two kernels driven by one DPS entry:
//   * problem_019_grad_x_launcher invoked TWICE — once for
//     (grad_q_embed, q, grad_q), once for (grad_k_embed, k, grad_k).
//   * problem_019_grad_emb_launcher invoked ONCE; per-(s) row CTA
//     accumulating per-(d) sums over the num_heads axis.
void problem_019_grad_x_launcher(
    const torch::Tensor& grad_x_embed,
    const torch::Tensor& embeddings,
    torch::Tensor& grad_x,
    cudaStream_t stream);

void problem_019_grad_emb_launcher(
    const torch::Tensor& grad_q_embed,
    const torch::Tensor& grad_k_embed,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& embeddings,
    torch::Tensor& grad_embeddings,
    cudaStream_t stream);

#endif  // PROBLEM_019_VISION_3D_ROPE_BACKWARD_KERNEL_H
