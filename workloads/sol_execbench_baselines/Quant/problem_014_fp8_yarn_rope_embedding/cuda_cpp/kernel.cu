#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

// Fused YaRN RoPE embedding with faithful FP8 e4m3 quantize-dequantize
// of the per-row frequency vector. The reference path:
//
//   freqs        = outer(position_ids, inv_freq)                   [S, 32]
//   freqs_pad    = pad_to_128(freqs)                                [S, 128]
//   scale        = max(amax(freqs_pad) / 448, 1e-12)                [S]
//   freqs_q      = clamp(freqs_pad / scale, -448, 448) -> e4m3 -> fp32
//   freqs_dq     = freqs_q * scale                                  [S, 128]
//   freqs_final  = freqs_dq[:, :32]                                 [S, 32]
//   emb          = concat([freqs_final, freqs_final], -1)           [S, 64]
//   cos_emb      = bf16(cos(emb) * mscale_combined)                 [S, 64]
//   sin_emb      = bf16(sin(emb) * mscale_combined)                 [S, 64]
//
// The padding zeros don't change the per-row amax (since amax is over
// absolute values and zero contributes 0). So we can compute amax over
// just the 32 real values and skip materialising the 128-wide padded
// vector entirely.
//
// One CTA per sequence position. The block uses 32 threads (one per
// frequency); a single warp's __shfl_xor reduction gives the per-row
// amax. Each thread also writes two output slots (slot k and slot
// k + 32) for cos_emb / sin_emb, exploiting the
// concat([freqs_final, freqs_final], -1) structure: emb[k] == emb[k+32]
// so cos / sin only depend on the lower-half index.

namespace {

constexpr int HALF_DIM = 32;
constexpr int FULL_DIM = 64;
constexpr float E4M3_MAX = 448.0f;
constexpr float SCALE_MIN = 1e-12f;
constexpr int BLOCK_THREADS = 32;

__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFFu, val, offset);
        if (other > val) val = other;
    }
    return val;
}

__global__ void __launch_bounds__(BLOCK_THREADS, 1)
problem_014_fp8_yarn_rope_embedding_kernel(
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ inv_freq,
    int seq_len,
    float mscale_combined,
    __nv_bfloat16* __restrict__ cos_emb,
    __nv_bfloat16* __restrict__ sin_emb) {

    const int row = blockIdx.x;
    if (row >= seq_len) return;
    const int tid = threadIdx.x;  // 0..31

    // Per-lane frequency: freq[tid] = position_ids[row] * inv_freq[tid].
    const float pos = static_cast<float>(position_ids[row]);
    const float freq = pos * inv_freq[tid];

    // Per-row amax across the 32 (real) frequency values via a single
    // warp shuffle reduction. The reference's padded zeros contribute
    // nothing to amax, so we can skip materialising the 128-wide
    // padded vector entirely.
    const float row_amax = warp_reduce_max(fabsf(freq));

    // Per-row FP8 scale derivation: max(amax / 448, 1e-12). The lower
    // clamp guards against a degenerate amax = 0 row producing inf.
    float scale = row_amax / E4M3_MAX;
    if (scale < SCALE_MIN) scale = SCALE_MIN;

    // Faithful FP8 quantize/dequantize per lane:
    //   freq_clamped = clamp(freq / scale, -E4M3_MAX, E4M3_MAX)
    //   fp8         = round_to_e4m3(freq_clamped)
    //   freq_dq     = fp32(fp8) * scale
    const float freq_div = freq / scale;
    const float freq_clamped = fmaxf(-E4M3_MAX, fminf(E4M3_MAX, freq_div));
    const __nv_fp8_e4m3 q = static_cast<__nv_fp8_e4m3>(freq_clamped);
    const float freq_q_fp32 = static_cast<float>(q);
    const float freq_final = freq_q_fp32 * scale;

    // Emit cos / sin to both halves of the 64-wide row. The reference
    // builds emb = concat([freqs_final, freqs_final], -1) so
    //   emb[row, tid]            = freq_final
    //   emb[row, tid + HALF_DIM] = freq_final
    // and cos / sin only depend on the lower-half index for both
    // output slots.
    const float c = cosf(freq_final) * mscale_combined;
    const float s = sinf(freq_final) * mscale_combined;
    const __nv_bfloat16 c_bf16 = __float2bfloat16_rn(c);
    const __nv_bfloat16 s_bf16 = __float2bfloat16_rn(s);

    const int slot0 = row * FULL_DIM + tid;
    const int slot1 = row * FULL_DIM + tid + HALF_DIM;
    cos_emb[slot0] = c_bf16;
    cos_emb[slot1] = c_bf16;
    sin_emb[slot0] = s_bf16;
    sin_emb[slot1] = s_bf16;
}

}  // namespace

void problem_014_fp8_yarn_rope_embedding_launcher(
    const torch::Tensor& position_ids,
    const float* inv_freq_host,
    float mscale_combined,
    torch::Tensor& cos_emb,
    torch::Tensor& sin_emb,
    cudaStream_t stream) {

    const int seq_len = static_cast<int>(position_ids.size(0));

    // Stage the 32 inv_freq values into device memory. We use a small
    // device buffer allocated per call; for the kernel's workload sizes
    // the 128B copy is negligible.
    float* inv_freq_dev = nullptr;
    cudaMallocAsync(&inv_freq_dev, HALF_DIM * sizeof(float), stream);
    cudaMemcpyAsync(
        inv_freq_dev, inv_freq_host, HALF_DIM * sizeof(float),
        cudaMemcpyHostToDevice, stream);

    dim3 grid(seq_len);
    dim3 block(BLOCK_THREADS);
    problem_014_fp8_yarn_rope_embedding_kernel<<<grid, block, 0, stream>>>(
        position_ids.data_ptr<int64_t>(),
        inv_freq_dev,
        seq_len,
        mscale_combined,
        reinterpret_cast<__nv_bfloat16*>(cos_emb.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(sin_emb.data_ptr<at::BFloat16>()));

    cudaFreeAsync(inv_freq_dev, stream);
}
