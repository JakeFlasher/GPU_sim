#include "kernel.h"
#include <cuda_runtime.h>

// fp32 RoPE application: rotate query/key tensors by precomputed
// cos/sin tables.
//
//   rotate_half(x)[..., d] = -x[..., d + half_head_dim]  for d < half_head_dim
//                          =  x[..., d - half_head_dim]  for d >= half_head_dim
//   x_rotated[..., d]      = x[..., d] * cos[s, d] +
//                            rotate_half(x)[..., d] * sin[s, d]
//
// head_dim = 128 (pinned by the problem definition); half_head_dim =
// 64. One CTA per (b, h, s) row; BLOCK_THREADS = head_dim so each
// thread handles exactly one element of the row. The cos and sin
// tables are indexed by (s, d) only, so they're reused across all
// (B, H) row CTAs.

namespace {

constexpr int kHeadDim = 128;
constexpr int kHalfHeadDim = 64;
constexpr int kBlockThreads = kHeadDim;

__global__ void __launch_bounds__(kBlockThreads, 8)
problem_088_rope_kernel(
    const float* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ x_rotated,
    int num_heads,
    int seq_len) {

    // Grid layout: (s, num_heads, B). Each CTA handles one
    // (b, h, s) row of head_dim elements.
    const int s = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int d = threadIdx.x;

    const int row_offset =
        ((b * num_heads + h) * seq_len + s) * kHeadDim;
    const int cs_offset = s * kHeadDim;

    const float xv = x[row_offset + d];

    // rotate_half partner:
    //   d < 64  -> partner = d + 64, sign = -1
    //   d >= 64 -> partner = d - 64, sign = +1
    int partner;
    float partner_sign;
    if (d < kHalfHeadDim) {
        partner = d + kHalfHeadDim;
        partner_sign = -1.0f;
    } else {
        partner = d - kHalfHeadDim;
        partner_sign = 1.0f;
    }
    const float xv_partner = x[row_offset + partner];
    const float rh = partner_sign * xv_partner;

    const float cs = cos_table[cs_offset + d];
    const float sn = sin_table[cs_offset + d];

    x_rotated[row_offset + d] = xv * cs + rh * sn;
}

}  // namespace

void problem_088_rotary_position_embedding_application_launcher(
    const torch::Tensor& x,
    const torch::Tensor& cos_table,
    const torch::Tensor& sin_table,
    torch::Tensor& x_rotated,
    cudaStream_t stream) {

    // x is [B, num_heads_or_kv, S, head_dim] fp32.
    const int batch_size = static_cast<int>(x.size(0));
    const int num_heads = static_cast<int>(x.size(1));
    const int seq_len = static_cast<int>(x.size(2));

    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(kBlockThreads);

    problem_088_rope_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        cos_table.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        x_rotated.data_ptr<float>(),
        num_heads,
        seq_len);
}
