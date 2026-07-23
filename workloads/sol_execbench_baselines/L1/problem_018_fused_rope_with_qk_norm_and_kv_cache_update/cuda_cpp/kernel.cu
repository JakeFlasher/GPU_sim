#include "kernel.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// head_dim is pinned at 128 per the problem definition's const axis.
// half_head_dim = 64. The RMS norm reduction over head_dim runs in a
// single warp (32 threads), so the kernel uses 128 threads per CTA
// (one thread per output element along head_dim) and a warp-shuffle
// reduction across each 32-element quarter, then a final reduction
// over the 4 quarter-warps via shared memory.
//
// One CTA per (batch, head, seq) row. Each thread handles one head_dim
// element. RoPE rotate_half uses shared memory to pair each thread's
// own normed value with the peer thread's (tid ± half_head_dim)
// normed value; the entire normed row lives in 128 BF16 values in
// shared memory, 256 B per CTA.

namespace {

constexpr int HEAD_DIM = 128;
constexpr int HALF_HEAD_DIM = 64;

// Reduce a per-thread fp32 value across one CTA of HEAD_DIM threads.
// Uses warp shuffle within each warp + shared memory across warps.
__device__ __forceinline__ float blockReduceSum_fp32(float val) {
    __shared__ float s_warp[4];  // HEAD_DIM / 32 = 4 warps.

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    // Warp-level reduction.
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    if (lane == 0) {
        s_warp[wid] = val;
    }
    __syncthreads();

    // Single-warp reduction across the 4 warp partial sums.
    val = (threadIdx.x < 4) ? s_warp[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffffu, val, offset);
        }
    }
    __shared__ float s_total;
    if (threadIdx.x == 0) {
        s_total = val;
    }
    __syncthreads();
    return s_total;
}

// Bit-clone every element of `src` into `dst` over the full cache shape.
// Launched with grid = (numel / 256, 1, 1), block = (256, 1, 1).
__global__ void copy_cache_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    long numel) {
    const long gid =
        static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid < numel) {
        dst[gid] = src[gid];
    }
}

// Apply RMS norm + RoPE to `in` (one row per CTA), write rotated bf16
// to `out`. The (batch, head, seq) for the input row is
// (blockIdx.z, blockIdx.y, blockIdx.x). num_heads is the head count for
// the specific tensor (96 for Q, 8 for K); the launcher passes the
// corresponding values.
__global__ void rms_rope_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ norm_weight,
    const long* __restrict__ position_ids,
    const float* __restrict__ inv_freq,
    float rms_norm_eps,
    __nv_bfloat16* __restrict__ out,
    int B, int num_heads, int seq_len) {

    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= B || h >= num_heads || s >= seq_len || tid >= HEAD_DIM) return;

    const long row_base =
        ((static_cast<long>(b) * num_heads + h) * seq_len + s) * HEAD_DIM;
    const long row_dst_base = row_base;
    const long pos_idx = static_cast<long>(b) * seq_len + s;

    // RMS norm: load BF16 → fp32, square, reduce across head_dim,
    // mean = sum / head_dim, inv_norm = rsqrtf(mean + eps).
    const float x = __bfloat162float(in[row_base + tid]);
    const float sq = x * x;
    const float sum_sq = blockReduceSum_fp32(sq);
    const float mean = sum_sq / static_cast<float>(HEAD_DIM);
    const float inv_norm = rsqrtf(mean + rms_norm_eps);

    // Apply weight (bf16) AFTER normalizing. The reference computes
    // `(weight.to(fp32) * x_normed).to(bf16)` so we stage the
    // multiplication in fp32 and cast at the end.
    const float w = __bfloat162float(norm_weight[tid]);
    const float normed_f = x * inv_norm * w;
    const __nv_bfloat16 normed_bf = __float2bfloat16(normed_f);

    // Stage the entire normed row in shared memory so each thread can
    // read its rotate-half peer (tid ± HALF_HEAD_DIM).
    __shared__ __nv_bfloat16 s_normed[HEAD_DIM];
    s_normed[tid] = normed_bf;
    __syncthreads();

    // Build cos / sin inline. inv_freq has half_head_dim entries; for
    // tid < half_head_dim the freq is inv_freq[tid] * pos_id; for
    // tid >= half_head_dim it is inv_freq[tid - half_head_dim] *
    // pos_id (the reference's `cat([freqs, freqs], dim=-1)`).
    const float pos_f = static_cast<float>(position_ids[pos_idx]);
    const int freq_idx = (tid < HALF_HEAD_DIM) ? tid : (tid - HALF_HEAD_DIM);
    const float freq = inv_freq[freq_idx] * pos_f;
    float cf, sf;
    __sincosf(freq, &sf, &cf);
    const __nv_bfloat16 cos_bf = __float2bfloat16(cf);
    const __nv_bfloat16 sin_bf = __float2bfloat16(sf);

    // rotate_half: peer index + sign.
    int peer_tid;
    float rotate_sign;
    if (tid < HALF_HEAD_DIM) {
        peer_tid = tid + HALF_HEAD_DIM;
        rotate_sign = -1.0f;
    } else {
        peer_tid = tid - HALF_HEAD_DIM;
        rotate_sign = 1.0f;
    }
    const float peer_f = rotate_sign * __bfloat162float(s_normed[peer_tid]);

    // Match pytorch bf16-mul: round each multiply result to bf16
    // before the add. See problem_071/cuda_cpp/kernel.cu for the
    // detailed rationale.
    const float own_f = __bfloat162float(normed_bf);
    const float cos_f = __bfloat162float(cos_bf);
    const float sin_f = __bfloat162float(sin_bf);
    const __nv_bfloat16 term1_bf = __float2bfloat16(own_f * cos_f);
    const __nv_bfloat16 term2_bf = __float2bfloat16(peer_f * sin_f);
    const float result =
        __bfloat162float(term1_bf) + __bfloat162float(term2_bf);
    out[row_dst_base + tid] = __float2bfloat16(result);
}

// Scatter the rotated K row into key_cache_out at the cache_position[s]
// slot. Grid (seq, num_kv_heads, batch), block 128 threads.
__global__ void scatter_k_kernel(
    const __nv_bfloat16* __restrict__ key_rotated,
    const long* __restrict__ cache_position,
    __nv_bfloat16* __restrict__ key_cache_out,
    int B, int num_kv_heads, int seq_len, int max_position_embeddings) {

    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= B || h >= num_kv_heads || s >= seq_len || tid >= HEAD_DIM) return;

    const long src_idx =
        ((static_cast<long>(b) * num_kv_heads + h) * seq_len + s) * HEAD_DIM + tid;
    const long pos = cache_position[s];
    const long dst_idx =
        ((static_cast<long>(b) * num_kv_heads + h) * max_position_embeddings + pos) * HEAD_DIM + tid;
    key_cache_out[dst_idx] = key_rotated[src_idx];
}

// Scatter raw V into value_cache_out at the cache_position[s] slot.
__global__ void scatter_v_kernel(
    const __nv_bfloat16* __restrict__ value,
    const long* __restrict__ cache_position,
    __nv_bfloat16* __restrict__ value_cache_out,
    int B, int num_kv_heads, int seq_len, int max_position_embeddings) {

    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= B || h >= num_kv_heads || s >= seq_len || tid >= HEAD_DIM) return;

    const long src_idx =
        ((static_cast<long>(b) * num_kv_heads + h) * seq_len + s) * HEAD_DIM + tid;
    const long pos = cache_position[s];
    const long dst_idx =
        ((static_cast<long>(b) * num_kv_heads + h) * max_position_embeddings + pos) * HEAD_DIM + tid;
    value_cache_out[dst_idx] = value[src_idx];
}

}  // namespace

void problem_018_fused_rope_with_qk_norm_and_kv_cache_update_launcher(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& position_ids,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& inv_freq,
    float rms_norm_eps,
    torch::Tensor& query_rotated,
    torch::Tensor& key_rotated,
    torch::Tensor& key_cache_out,
    torch::Tensor& value_cache_out,
    cudaStream_t stream) {

    const int B = static_cast<int>(query.size(0));
    const int num_q_heads = static_cast<int>(query.size(1));
    const int seq_len = static_cast<int>(query.size(2));
    const int num_kv_heads = static_cast<int>(key.size(1));
    const int max_position_embeddings = static_cast<int>(key_cache.size(2));

    // Clone the cache.
    const long cache_numel = key_cache.numel();
    cudaMemcpyAsync(
        key_cache_out.data_ptr<at::BFloat16>(),
        key_cache.data_ptr<at::BFloat16>(),
        cache_numel * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice,
        stream);
    cudaMemcpyAsync(
        value_cache_out.data_ptr<at::BFloat16>(),
        value_cache.data_ptr<at::BFloat16>(),
        cache_numel * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice,
        stream);

    if (seq_len == 0) {
        return;
    }

    // Q: RMS norm + RoPE.
    {
        dim3 grid(seq_len, num_q_heads, B);
        dim3 block(HEAD_DIM);
        rms_rope_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                q_norm_weight.data_ptr<at::BFloat16>()),
            position_ids.data_ptr<long>(),
            inv_freq.data_ptr<float>(),
            rms_norm_eps,
            reinterpret_cast<__nv_bfloat16*>(
                query_rotated.data_ptr<at::BFloat16>()),
            B, num_q_heads, seq_len);
    }

    // K: RMS norm + RoPE.
    {
        dim3 grid(seq_len, num_kv_heads, B);
        dim3 block(HEAD_DIM);
        rms_rope_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                k_norm_weight.data_ptr<at::BFloat16>()),
            position_ids.data_ptr<long>(),
            inv_freq.data_ptr<float>(),
            rms_norm_eps,
            reinterpret_cast<__nv_bfloat16*>(
                key_rotated.data_ptr<at::BFloat16>()),
            B, num_kv_heads, seq_len);
    }

    // Scatter rotated K and raw V into the cloned caches.
    {
        dim3 grid(seq_len, num_kv_heads, B);
        dim3 block(HEAD_DIM);
        scatter_k_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                key_rotated.data_ptr<at::BFloat16>()),
            cache_position.data_ptr<long>(),
            reinterpret_cast<__nv_bfloat16*>(
                key_cache_out.data_ptr<at::BFloat16>()),
            B, num_kv_heads, seq_len, max_position_embeddings);

        scatter_v_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
            cache_position.data_ptr<long>(),
            reinterpret_cast<__nv_bfloat16*>(
                value_cache_out.data_ptr<at::BFloat16>()),
            B, num_kv_heads, seq_len, max_position_embeddings);
    }
}
