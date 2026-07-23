#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// One CTA per (batch, seq) pair. half_head_dim threads cooperatively
// compute the cos/sin pair for one position. The second half of the
// concatenation `emb = cat(freqs, freqs)` is bit-identical to the
// first, so each thread fans its (cos, sin) pair to TWO output slots
// at offsets `tid` and `tid + half_head_dim` along head_dim.
//
// inv_freq[half_head_dim] is small (64 fp32 = 256 B) and identical
// across all blocks; we stage it into shared memory once per CTA so
// the per-row loop does not re-fetch from global memory.

namespace {

constexpr int HEAD_DIM = 128;
constexpr int HALF_HEAD_DIM = 64;

__global__ void __launch_bounds__(HALF_HEAD_DIM, 16)
problem_011_rotary_position_embedding_kernel(
    const long* __restrict__ position_ids,
    const float* __restrict__ inv_freq,
    float attention_scaling,
    __nv_bfloat16* __restrict__ cos_sin,
    int batch_size,
    int seq_len) {

    const int batch_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    __shared__ float s_inv_freq[HALF_HEAD_DIM];
    const int tid = threadIdx.x;
    if (tid < HALF_HEAD_DIM) {
        s_inv_freq[tid] = inv_freq[tid];
    }
    __syncthreads();

    const long pos = position_ids[batch_idx * seq_len + seq_idx];
    const float pos_f = static_cast<float>(pos);

    const float freq = s_inv_freq[tid] * pos_f;
    float c, s;
    __sincosf(freq, &s, &c);
    const float cos_val = c * attention_scaling;
    const float sin_val = s * attention_scaling;

    // cos_sin layout: [batch, seq, head_dim, 2] contiguous.
    // Index of (batch_idx, seq_idx, d, k) is
    //   ((batch_idx * seq_len + seq_idx) * HEAD_DIM + d) * 2 + k.
    const long row_base =
        (static_cast<long>(batch_idx) * seq_len + seq_idx) *
        (HEAD_DIM * 2);

    const __nv_bfloat16 cos_bf = __float2bfloat16(cos_val);
    const __nv_bfloat16 sin_bf = __float2bfloat16(sin_val);

    // First-half slot: d = tid.
    cos_sin[row_base + tid * 2 + 0] = cos_bf;
    cos_sin[row_base + tid * 2 + 1] = sin_bf;
    // Second-half slot: d = tid + HALF_HEAD_DIM. emb is concat(f, f) so
    // the cos/sin pair is identical to the first-half slot.
    cos_sin[row_base + (tid + HALF_HEAD_DIM) * 2 + 0] = cos_bf;
    cos_sin[row_base + (tid + HALF_HEAD_DIM) * 2 + 1] = sin_bf;
}

}  // namespace

void problem_011_rotary_position_embedding_launcher(
    const torch::Tensor& position_ids,
    const torch::Tensor& inv_freq,
    float attention_scaling,
    torch::Tensor& cos_sin,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(position_ids.size(0));
    const int seq_len = static_cast<int>(position_ids.size(1));

    dim3 grid(seq_len, batch_size);
    dim3 block(HALF_HEAD_DIM);

    problem_011_rotary_position_embedding_kernel<<<grid, block, 0, stream>>>(
        position_ids.data_ptr<long>(),
        inv_freq.data_ptr<float>(),
        attention_scaling,
        reinterpret_cast<__nv_bfloat16*>(cos_sin.data_ptr<at::BFloat16>()),
        batch_size,
        seq_len);
}
