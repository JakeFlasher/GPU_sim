#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// One thread per (head_dim) output position; one block per (batch,
// seq) row. The reference performs emb = cat(freqs, freqs) along
// the last dim then sin/cos with attention scaling. We fold the
// replication into the index math: for lane d in [0, head_dim),
// the input index is d % half_head_dim, so lanes [0, half) and
// [half, head_dim) read freqs at the same position. Each thread
// emits one cos element and one sin element.

namespace {

constexpr int BLOCK_THREADS = 128;  // head_dim baked at the spec value

__global__ void problem_012_fused_cos_sin_kernel(
    const float* __restrict__ freqs,
    __nv_bfloat16* __restrict__ cos_out,
    __nv_bfloat16* __restrict__ sin_out,
    int row_count,
    int half_head_dim,
    int head_dim,
    float attention_scaling) {

    const int row = blockIdx.x;
    if (row >= row_count) {
        return;
    }
    const int d = threadIdx.x;
    if (d >= head_dim) {
        return;
    }

    const int freq_idx = d < half_head_dim ? d : d - half_head_dim;
    const float x = freqs[row * half_head_dim + freq_idx];

    float c, s;
    __sincosf(x, &s, &c);
    c *= attention_scaling;
    s *= attention_scaling;

    const int out_offset = row * head_dim + d;
    cos_out[out_offset] = __float2bfloat16_rn(c);
    sin_out[out_offset] = __float2bfloat16_rn(s);
}

}  // namespace

void problem_012_fused_cos_sin_launcher(
    const torch::Tensor& freqs,
    float attention_scaling,
    torch::Tensor& cos_out,
    torch::Tensor& sin_out,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(freqs.size(0));
    const int seq_len = static_cast<int>(freqs.size(1));
    const int half_head_dim = static_cast<int>(freqs.size(2));
    const int head_dim = static_cast<int>(cos_out.size(2));
    const int row_count = batch_size * seq_len;

    dim3 grid(row_count);
    dim3 block(BLOCK_THREADS);

    problem_012_fused_cos_sin_kernel<<<grid, block, 0, stream>>>(
        freqs.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(cos_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(sin_out.data_ptr<at::BFloat16>()),
        row_count,
        half_head_dim,
        head_dim,
        attention_scaling);
}
