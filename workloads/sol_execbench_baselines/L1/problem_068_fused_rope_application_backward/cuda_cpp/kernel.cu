#include "kernel.h"
#include <cuda_runtime.h>

// fp32 fused RoPE application backward.
//
// Kernel A computes grad_x = grad_x_embed * cos - rotate_half(grad_x_embed)
// * sin per element. Grid: (seq_len, num_heads_axis, batch_size).
// BLOCK_THREADS = head_dim = 128.
//
// Kernel B computes grad_cos[b, s, d] and grad_sin[b, s, d] as the
// per-head sum of (grad_q_embed * q) and (grad_q_embed * rotate_half(q))
// plus the same K-side contributions. Grid: (seq_len, batch_size).
// BLOCK_THREADS = head_dim = 128. Each thread holds two fp32
// accumulators and iterates over the head axis in a stride-loop.

namespace {

constexpr int kHeadDim = 128;
constexpr int kHalfHeadDim = 64;
constexpr int kBlockThreads = kHeadDim;

__device__ inline void resolve_partner(
    int d, int& partner_d, float& partner_sign) {
    if (d < kHalfHeadDim) {
        partner_d = d + kHalfHeadDim;
        partner_sign = -1.0f;
    } else {
        partner_d = d - kHalfHeadDim;
        partner_sign = 1.0f;
    }
}

__global__ void __launch_bounds__(kBlockThreads, 8)
problem_068_grad_x_kernel(
    const float* __restrict__ grad_x_embed,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ grad_x,
    int num_heads_axis,
    int seq_len) {

    const int s = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int d = threadIdx.x;

    int partner_d;
    float partner_sign;
    resolve_partner(d, partner_d, partner_sign);

    const int row_offset =
        ((b * num_heads_axis + h) * seq_len + s) * kHeadDim;
    const int cs_offset = (b * seq_len + s) * kHeadDim;

    const float g       = grad_x_embed[row_offset + d];
    const float g_partner = grad_x_embed[row_offset + partner_d];
    const float cs      = cos_table[cs_offset + d];
    const float sn      = sin_table[cs_offset + d];

    // grad_x = grad_output * cos - rotate_half(grad_output) * sin
    //        = g * cs - (partner_sign * g_partner) * sn
    grad_x[row_offset + d] = g * cs - partner_sign * g_partner * sn;
}

__global__ void __launch_bounds__(kBlockThreads, 4)
problem_068_grad_cos_sin_kernel(
    const float* __restrict__ grad_q_embed,
    const float* __restrict__ grad_k_embed,
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ grad_cos,
    float* __restrict__ grad_sin,
    int num_heads,
    int num_kv_heads,
    int seq_len) {

    const int s = blockIdx.x;
    const int b = blockIdx.y;
    const int d = threadIdx.x;

    int partner_d;
    float partner_sign;
    resolve_partner(d, partner_d, partner_sign);

    float acc_cos = 0.0f;
    float acc_sin = 0.0f;

    // Q-side reduction over num_heads heads.
    for (int h = 0; h < num_heads; ++h) {
        const int row_offset =
            ((b * num_heads + h) * seq_len + s) * kHeadDim;
        const float g     = grad_q_embed[row_offset + d];
        const float x_v   = q[row_offset + d];
        const float x_par = q[row_offset + partner_d];
        const float rh    = partner_sign * x_par;
        acc_cos += g * x_v;
        acc_sin += g * rh;
    }

    // K-side reduction over num_kv_heads heads.
    for (int h = 0; h < num_kv_heads; ++h) {
        const int row_offset =
            ((b * num_kv_heads + h) * seq_len + s) * kHeadDim;
        const float g     = grad_k_embed[row_offset + d];
        const float x_v   = k[row_offset + d];
        const float x_par = k[row_offset + partner_d];
        const float rh    = partner_sign * x_par;
        acc_cos += g * x_v;
        acc_sin += g * rh;
    }

    const int cs_offset = (b * seq_len + s) * kHeadDim;
    grad_cos[cs_offset + d] = acc_cos;
    grad_sin[cs_offset + d] = acc_sin;
}

}  // namespace

void problem_068_grad_x_launcher(
    const torch::Tensor& grad_x_embed,
    const torch::Tensor& cos_table,
    const torch::Tensor& sin_table,
    torch::Tensor& grad_x,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(grad_x_embed.size(0));
    const int num_heads_axis = static_cast<int>(grad_x_embed.size(1));
    const int seq_len = static_cast<int>(grad_x_embed.size(2));

    dim3 grid(seq_len, num_heads_axis, batch_size);
    dim3 block(kBlockThreads);

    problem_068_grad_x_kernel<<<grid, block, 0, stream>>>(
        grad_x_embed.data_ptr<float>(),
        cos_table.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        num_heads_axis,
        seq_len);
}

void problem_068_grad_cos_sin_launcher(
    const torch::Tensor& grad_q_embed,
    const torch::Tensor& grad_k_embed,
    const torch::Tensor& q,
    const torch::Tensor& k,
    torch::Tensor& grad_cos,
    torch::Tensor& grad_sin,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(grad_q_embed.size(0));
    const int num_heads = static_cast<int>(grad_q_embed.size(1));
    const int num_kv_heads = static_cast<int>(grad_k_embed.size(1));
    const int seq_len = static_cast<int>(grad_q_embed.size(2));

    dim3 grid(seq_len, batch_size);
    dim3 block(kBlockThreads);

    problem_068_grad_cos_sin_kernel<<<grid, block, 0, stream>>>(
        grad_q_embed.data_ptr<float>(),
        grad_k_embed.data_ptr<float>(),
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        grad_cos.data_ptr<float>(),
        grad_sin.data_ptr<float>(),
        num_heads,
        num_kv_heads,
        seq_len);
}
