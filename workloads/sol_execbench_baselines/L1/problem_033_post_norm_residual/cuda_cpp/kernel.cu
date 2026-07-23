#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// BF16 post-norm residual: output = residual + RMSNorm(sublayer_output, weight, eps).
//
// Per the Olmo3 reference, the RMSNorm step runs in fp32 (the bf16
// inputs upcast to fp32 for the variance + rescale, then downcast
// to bf16 before the residual add). The residual add is therefore
// a pure bf16 + bf16 operation — distinct from the
// `problem_001/002/003_fused_add_rmsnorm` family where the
// residual is mixed in BEFORE the normalization.
//
// One CTA per row; BLOCK_THREADS=256 stride-loops over the
// `HIDDEN_SIZE_VEC = HIDDEN_SIZE / 2 = 2048` BF16x2 pairs per
// row. fp32 sum-of-squares reduction via warp shuffle +
// cross-warp shared-memory tree (kMaxWarps=32). The eps is a
// runtime fp32 scalar (the reference signature uses
// `eps: float`).

namespace {

constexpr int HIDDEN_SIZE = 4096;
constexpr int HIDDEN_SIZE_VEC = 2048;
constexpr int BLOCK_THREADS = 256;
constexpr int MAX_WARPS = 32;

__device__ inline float warp_all_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

__global__ void __launch_bounds__(BLOCK_THREADS, 4)
problem_033_post_norm_residual_kernel(
    const __nv_bfloat16* __restrict__ sublayer_output,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    float eps,
    __nv_bfloat16* __restrict__ output,
    int n_rows) {

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const __nv_bfloat162* x_row =
        reinterpret_cast<const __nv_bfloat162*>(sublayer_output)
        + row * HIDDEN_SIZE_VEC;
    const __nv_bfloat162* r_row =
        reinterpret_cast<const __nv_bfloat162*>(residual)
        + row * HIDDEN_SIZE_VEC;
    const __nv_bfloat162* w_vec =
        reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* out_row =
        reinterpret_cast<__nv_bfloat162*>(output) + row * HIDDEN_SIZE_VEC;

    const int tid = threadIdx.x;

    // Pass 1: sum-of-squares of sublayer_output in fp32.
    float thread_sum_sq = 0.0f;
    #pragma unroll 1
    for (int i = tid; i < HIDDEN_SIZE_VEC; i += BLOCK_THREADS) {
        __nv_bfloat162 xv = __ldg(x_row + i);
        float2 x_f2 = __bfloat1622float2(xv);
        thread_sum_sq += x_f2.x * x_f2.x + x_f2.y * x_f2.y;
    }

    thread_sum_sq = warp_all_reduce_sum(thread_sum_sq);

    __shared__ float s_warp[MAX_WARPS];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = (BLOCK_THREADS + 31) >> 5;
    if (lane == 0) s_warp[warp] = thread_sum_sq;
    __syncthreads();

    if (warp == 0) {
        float partial = (lane < num_warps) ? s_warp[lane] : 0.0f;
        partial = warp_all_reduce_sum(partial);
        if (lane == 0) s_warp[0] = partial;
    }
    __syncthreads();
    const float row_sum = s_warp[0];

    const float inv_rms = rsqrtf(
        row_sum / static_cast<float>(HIDDEN_SIZE) + eps);

    // Pass 2: re-load sublayer_output + residual + weight, write
    // residual-added RMSNorm output as bf16.
    #pragma unroll 1
    for (int i = tid; i < HIDDEN_SIZE_VEC; i += BLOCK_THREADS) {
        __nv_bfloat162 xv = __ldg(x_row + i);
        __nv_bfloat162 rv = __ldg(r_row + i);
        __nv_bfloat162 wv = __ldg(w_vec + i);
        float2 x_f2 = __bfloat1622float2(xv);
        float2 w_f2 = __bfloat1622float2(wv);
        // Normalize + scale in fp32, then downcast to bf16
        // (matching the reference's `.to(input_dtype)` step
        // BEFORE the residual add).
        float n0 = x_f2.x * inv_rms * w_f2.x;
        float n1 = x_f2.y * inv_rms * w_f2.y;
        __nv_bfloat162 normalized = __float22bfloat162_rn(
            make_float2(n0, n1));
        // Residual add in bf16.
        __nv_bfloat162 y = __hadd2(rv, normalized);
        out_row[i] = y;
    }
}

}  // namespace

void problem_033_post_norm_residual_launcher(
    const torch::Tensor& sublayer_output,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    float eps,
    torch::Tensor& output,
    cudaStream_t stream) {

    // sublayer_output and residual are [B, S, H] bf16; one CTA per
    // row over B*S.
    const int n_rows = static_cast<int>(
        sublayer_output.size(0) * sublayer_output.size(1));

    dim3 grid(n_rows);
    dim3 block(BLOCK_THREADS);

    problem_033_post_norm_residual_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            sublayer_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            weight.data_ptr<at::BFloat16>()),
        eps,
        reinterpret_cast<__nv_bfloat16*>(
            output.data_ptr<at::BFloat16>()),
        n_rows);
}
