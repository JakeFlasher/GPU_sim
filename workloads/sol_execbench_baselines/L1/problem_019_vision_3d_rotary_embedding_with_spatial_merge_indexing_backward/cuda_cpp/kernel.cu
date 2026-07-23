#include "kernel.h"
#include <cuda_runtime.h>

// fp32 Vision 3D RoPE backward.
//
// Kernel A computes grad_q or grad_k per element using the
// rotate-half-backward pattern. Cos/sin tables are derived
// in-kernel from the `embeddings[seq_len, head_dim]` tensor.
//
// Kernel B accumulates grad_embeddings[s, d] by summing
// per-head contributions of (grad_q_embed * q) + (grad_k_embed * k)
// for the cos-derivative branch, and the rotated-h variants for
// the sin-derivative branch, then mixes them via cos(emb) and
// sin(emb).

namespace {

constexpr int kHeadDim = 128;
constexpr int kHalfHeadDim = 64;
constexpr int kBlockThreads = kHeadDim;
constexpr int kReductionThreads = 256;

__device__ inline void resolve_partner(
    int d, int& partner_d, float& partner_sign_fwd) {
    if (d < kHalfHeadDim) {
        partner_d = d + kHalfHeadDim;
        partner_sign_fwd = -1.0f;
    } else {
        partner_d = d - kHalfHeadDim;
        partner_sign_fwd = 1.0f;
    }
}

__global__ void __launch_bounds__(kBlockThreads, 8)
problem_019_grad_x_kernel(
    const float* __restrict__ grad_x_embed,
    const float* __restrict__ embeddings,
    float* __restrict__ grad_x,
    int num_heads,
    int seq_len) {

    // Grid: (num_heads, seq_len). One CTA per (s, h) row.
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int d = threadIdx.x;

    int partner_d;
    float partner_sign_fwd;
    resolve_partner(d, partner_d, partner_sign_fwd);

    // Row offsets:
    //   grad_x_embed and grad_x: [S, H, D] indexed as
    //     (s*num_heads + h)*head_dim + d
    //   embeddings: [S, D] indexed as s*head_dim + d
    const int row_offset = (s * num_heads + h) * kHeadDim;
    const int emb_offset = s * kHeadDim;

    const float g_d       = grad_x_embed[row_offset + d];
    const float g_partner = grad_x_embed[row_offset + partner_d];
    const float emb_d     = embeddings[emb_offset + d];
    const float emb_par   = embeddings[emb_offset + partner_d];
    const float cos_d     = cosf(emb_d);
    const float sin_par   = sinf(emb_par);

    grad_x[row_offset + d] =
        g_d * cos_d - partner_sign_fwd * g_partner * sin_par;
}

__global__ void __launch_bounds__(kReductionThreads, 4)
problem_019_grad_emb_kernel(
    const float* __restrict__ grad_q_embed,
    const float* __restrict__ grad_k_embed,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ embeddings,
    float* __restrict__ grad_embeddings,
    int num_heads,
    int seq_len) {

    // Grid: (seq_len,). One CTA per s. BLOCK_THREADS = 256 stride
    // loops over head_dim = 128.
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    const int emb_offset = s * kHeadDim;

    for (int d = tid; d < kHeadDim; d += kReductionThreads) {
        int partner_d;
        float partner_sign_fwd;
        resolve_partner(d, partner_d, partner_sign_fwd);

        float acc_cos = 0.0f;
        float acc_sin = 0.0f;
        for (int h = 0; h < num_heads; ++h) {
            const int qk_row_offset =
                (s * num_heads + h) * kHeadDim;
            const float g_q      = grad_q_embed[qk_row_offset + d];
            const float g_k      = grad_k_embed[qk_row_offset + d];
            const float q_d      = q[qk_row_offset + d];
            const float k_d      = k[qk_row_offset + d];
            const float q_par    = q[qk_row_offset + partner_d];
            const float k_par    = k[qk_row_offset + partner_d];
            acc_cos += g_q * q_d + g_k * k_d;
            acc_sin += g_q * q_par + g_k * k_par;
        }
        // grad_sin_total = partner_sign_fwd * Σ_h(g * x_partner_via_rotate_half)
        acc_sin *= partner_sign_fwd;

        const float emb_d   = embeddings[emb_offset + d];
        const float cos_e_d = cosf(emb_d);
        const float sin_e_d = sinf(emb_d);
        grad_embeddings[emb_offset + d] =
            -acc_cos * sin_e_d + acc_sin * cos_e_d;
    }
}

}  // namespace

void problem_019_grad_x_launcher(
    const torch::Tensor& grad_x_embed,
    const torch::Tensor& embeddings,
    torch::Tensor& grad_x,
    cudaStream_t stream) {

    const int seq_len = static_cast<int>(grad_x_embed.size(0));
    const int num_heads = static_cast<int>(grad_x_embed.size(1));

    dim3 grid(num_heads, seq_len);
    dim3 block(kBlockThreads);

    problem_019_grad_x_kernel<<<grid, block, 0, stream>>>(
        grad_x_embed.data_ptr<float>(),
        embeddings.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        num_heads,
        seq_len);
}

void problem_019_grad_emb_launcher(
    const torch::Tensor& grad_q_embed,
    const torch::Tensor& grad_k_embed,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& embeddings,
    torch::Tensor& grad_embeddings,
    cudaStream_t stream) {

    const int seq_len = static_cast<int>(grad_q_embed.size(0));
    const int num_heads = static_cast<int>(grad_q_embed.size(1));

    dim3 grid(seq_len);
    dim3 block(kReductionThreads);

    problem_019_grad_emb_kernel<<<grid, block, 0, stream>>>(
        grad_q_embed.data_ptr<float>(),
        grad_k_embed.data_ptr<float>(),
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        embeddings.data_ptr<float>(),
        grad_embeddings.data_ptr<float>(),
        num_heads,
        seq_len);
}
