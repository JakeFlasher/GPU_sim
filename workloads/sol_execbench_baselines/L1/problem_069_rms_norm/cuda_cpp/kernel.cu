#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// One CTA per (batch, seq) row. The kernel:
//   1. Reads (hidden_states + residual) into BF16x2 pairs, accumulating
//      the per-row sum-of-squares in FP32.
//   2. Reduces sum-of-squares across the block via warp shuffle +
//      shared-memory inter-warp reduction.
//   3. Computes inv_rms = rsqrtf(mean_sq + eps) in FP32.
//   4. Rewrites output[row, :] = ((h + r) * inv_rms * weight) cast back
//      to BF16. The multiplication by `weight` is folded into the FP32
//      path before the BF16 cast; the precision contract requires FP32
//      accumulation and tolerates this fused form (the reference's
//      separate BF16 multiply is within tolerance).
//
// hidden_size = 8192 is baked in as a constexpr (per definition.json:
// "hidden_size" is a const axis with value 8192). The 3D row layout
// flattens to row_count = batch_size * seq_len; per-row stride is
// hidden_size BF16 elements (the tensor is contiguous on entry).

namespace {

__device__ inline float warp_all_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

constexpr int HIDDEN_SIZE = 8192;
constexpr int HIDDEN_SIZE_VEC = 4096;  // BF16x2 pairs per row
constexpr int BLOCK_THREADS = 1024;
constexpr int MAX_WARPS = 32;

__global__ void __launch_bounds__(BLOCK_THREADS, 1) problem_069_rms_norm_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int row_count,
    float eps) {

    const int row = blockIdx.x;
    if (row >= row_count) {
        return;
    }

    const __nv_bfloat162* hs_row =
        reinterpret_cast<const __nv_bfloat162*>(hidden_states) + row * HIDDEN_SIZE_VEC;
    const __nv_bfloat162* res_row =
        reinterpret_cast<const __nv_bfloat162*>(residual) + row * HIDDEN_SIZE_VEC;
    const __nv_bfloat162* w_vec =
        reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* out_row =
        reinterpret_cast<__nv_bfloat162*>(output) + row * HIDDEN_SIZE_VEC;

    const int tid = threadIdx.x;

    float thread_sum_sq = 0.0f;
    #pragma unroll 1
    for (int i = tid; i < HIDDEN_SIZE_VEC; i += BLOCK_THREADS) {
        __nv_bfloat162 hv = __ldg(hs_row + i);
        __nv_bfloat162 rv = __ldg(res_row + i);
        float2 h_f2 = __bfloat1622float2(hv);
        float2 r_f2 = __bfloat1622float2(rv);
        float x0 = h_f2.x + r_f2.x;
        float x1 = h_f2.y + r_f2.y;
        thread_sum_sq += x0 * x0 + x1 * x1;
    }

    thread_sum_sq = warp_all_reduce_sum(thread_sum_sq);

    __shared__ float s_warp[MAX_WARPS];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = (BLOCK_THREADS + 31) >> 5;
    if (lane == 0) {
        s_warp[warp] = thread_sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float partial = (lane < num_warps) ? s_warp[lane] : 0.0f;
        partial = warp_all_reduce_sum(partial);
        if (lane == 0) {
            s_warp[0] = partial;
        }
    }
    __syncthreads();
    const float row_sum = s_warp[0];

    const float inv_rms = rsqrtf(row_sum / static_cast<float>(HIDDEN_SIZE) + eps);

    #pragma unroll 1
    for (int i = tid; i < HIDDEN_SIZE_VEC; i += BLOCK_THREADS) {
        __nv_bfloat162 hv = __ldg(hs_row + i);
        __nv_bfloat162 rv = __ldg(res_row + i);
        __nv_bfloat162 wv = __ldg(w_vec + i);
        float2 h_f2 = __bfloat1622float2(hv);
        float2 r_f2 = __bfloat1622float2(rv);
        float2 w_f2 = __bfloat1622float2(wv);
        float y0 = (h_f2.x + r_f2.x) * inv_rms * w_f2.x;
        float y1 = (h_f2.y + r_f2.y) * inv_rms * w_f2.y;
        out_row[i] = __float22bfloat162_rn(make_float2(y0, y1));
    }
}

}  // namespace

void problem_069_rms_norm_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    float eps,
    torch::Tensor& output,
    cudaStream_t stream) {

    // hidden_states is [batch_size, seq_len, hidden_size] with
    // hidden_size known at compile time. Flatten the leading two
    // dimensions into a single row count and dispatch one CTA per row.
    const int batch_size = static_cast<int>(hidden_states.size(0));
    const int seq_len = static_cast<int>(hidden_states.size(1));
    const int row_count = batch_size * seq_len;
    dim3 grid(row_count);
    dim3 block(BLOCK_THREADS);

    problem_069_rms_norm_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        row_count,
        eps);
}
