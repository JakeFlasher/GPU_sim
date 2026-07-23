#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// BF16 attention softmax with Gemma3 logit softcapping.
//
// Per the problem definition, the reference does:
//   softcapped = tanh(attn_weights / 30) * 30   # element-wise softcap
//   output     = softmax(softcapped, dim=-1, dtype=fp32).to(bf16)
//
// One CTA per row (B*H*Q rows). Threads cooperatively:
//   pass 1: load x, apply softcap, reduce row max in fp32
//   pass 2: load x, apply softcap, reduce sum of exp(softcap - max)
//   pass 3: load x, apply softcap, write exp(softcap - max) / sum as bf16
//
// The softcap (tanh + 2 muls) is recomputed in each pass rather
// than cached in shared memory; for typical seq_len_k (the
// workload sweep covers seq_len_k up to ~1024) the recompute
// cost is dominated by the bf16 global load, and avoiding the
// shared-memory cache keeps occupancy high.

namespace {

constexpr float kSoftcap = 30.0f;
constexpr float kInvSoftcap = 1.0f / kSoftcap;  // 1/30

__device__ __forceinline__ float softcap_f(float x) {
    return tanhf(x * kInvSoftcap) * kSoftcap;
}

__device__ inline float warp_all_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
    }
    return val;
}

__device__ inline float warp_all_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

constexpr int kBlockThreads = 256;
constexpr int kMaxWarps = 32;

__global__ void __launch_bounds__(kBlockThreads, 4)
problem_046_attention_softmax_softcap_kernel(
    const __nv_bfloat16* __restrict__ attn_weights,
    __nv_bfloat16* __restrict__ output,
    int seq_len_k) {

    const int row = blockIdx.x;
    const __nv_bfloat16* xr = attn_weights + (size_t)row * seq_len_k;
    __nv_bfloat16* yr = output + (size_t)row * seq_len_k;
    const int tid = threadIdx.x;

    // Pass 1: find row max of softcap(x).
    float thread_max = -1e30f;
    #pragma unroll 1
    for (int k = tid; k < seq_len_k; k += kBlockThreads) {
        float x = __bfloat162float(__ldg(xr + k));
        thread_max = fmaxf(thread_max, softcap_f(x));
    }

    thread_max = warp_all_reduce_max(thread_max);

    __shared__ float s_warp[kMaxWarps];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = (kBlockThreads + 31) >> 5;
    if (lane == 0) s_warp[warp] = thread_max;
    __syncthreads();

    if (warp == 0) {
        float partial = (lane < num_warps) ? s_warp[lane] : -1e30f;
        partial = warp_all_reduce_max(partial);
        if (lane == 0) s_warp[0] = partial;
    }
    __syncthreads();
    const float row_max = s_warp[0];

    // Pass 2: sum of exp(softcap(x) - row_max).
    float thread_sum = 0.0f;
    #pragma unroll 1
    for (int k = tid; k < seq_len_k; k += kBlockThreads) {
        float x = __bfloat162float(__ldg(xr + k));
        thread_sum += __expf(softcap_f(x) - row_max);
    }

    thread_sum = warp_all_reduce_sum(thread_sum);
    if (lane == 0) s_warp[warp] = thread_sum;
    __syncthreads();

    if (warp == 0) {
        float partial = (lane < num_warps) ? s_warp[lane] : 0.0f;
        partial = warp_all_reduce_sum(partial);
        if (lane == 0) s_warp[0] = partial;
    }
    __syncthreads();
    const float row_sum = s_warp[0];
    const float inv_row_sum = 1.0f / row_sum;

    // Pass 3: write exp(softcap(x) - row_max) / row_sum as bf16.
    #pragma unroll 1
    for (int k = tid; k < seq_len_k; k += kBlockThreads) {
        float x = __bfloat162float(__ldg(xr + k));
        float y = __expf(softcap_f(x) - row_max) * inv_row_sum;
        yr[k] = __float2bfloat16(y);
    }
}

}  // namespace

void problem_046_attention_softmax_softcap_launcher(
    const torch::Tensor& attn_weights,
    torch::Tensor& output,
    cudaStream_t stream) {

    // attn_weights is [B, H, Q, K] bf16; one CTA per row over B*H*Q.
    const int seq_len_k = static_cast<int>(attn_weights.size(3));
    const int n_rows = static_cast<int>(
        attn_weights.size(0) * attn_weights.size(1) * attn_weights.size(2));

    dim3 grid(n_rows);
    dim3 block(kBlockThreads);

    problem_046_attention_softmax_softcap_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(attn_weights.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        seq_len_k);
}
