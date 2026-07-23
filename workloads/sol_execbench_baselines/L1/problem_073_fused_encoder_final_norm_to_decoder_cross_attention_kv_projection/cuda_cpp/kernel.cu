#include "kernel.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// One CTA per (batch, seq) position; total batch*seq CTAs.
// Per CTA: cooperative RMSNorm over hidden_size=1024 (FP32
// accumulator, eps-stabilized rsqrt) then writes the FP16-quantised
// normalized vector to the output tensor in FP32→FP16 round-to-
// nearest semantics matching the reference's
//     normalized = (norm_weight * hidden_states).to(input_dtype)
// step. The K/V projection matmuls are NOT done here; the host
// caller delegates them to `at::matmul` so the reference's reduced-
// precision cuBLAS HGEMM path is honoured exactly.

namespace {

__device__ inline float warp_all_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

constexpr int HIDDEN_SIZE = 1024;        // pinned by definition.json
constexpr int BLOCK_THREADS = 128;
constexpr int NUM_WARPS = BLOCK_THREADS / 32;  // 4

__global__ void __launch_bounds__(BLOCK_THREADS, 4)
problem_073_rmsnorm_kernel(
    const __half* __restrict__ encoder_hidden_states,
    const __half* __restrict__ norm_weight,
    __half* __restrict__ normalized,
    int batch_size,
    int seq_len,
    float eps) {

    const int row_idx = blockIdx.x;
    const int total_rows = batch_size * seq_len;
    if (row_idx >= total_rows) return;

    const __half* hs_row =
        encoder_hidden_states + row_idx * HIDDEN_SIZE;
    __half* norm_row = normalized + row_idx * HIDDEN_SIZE;
    const int tid = threadIdx.x;

    // Phase 1: sum of squares for RMSNorm.
    float thread_sum_sq = 0.0f;
    #pragma unroll
    for (int i = tid; i < HIDDEN_SIZE; i += BLOCK_THREADS) {
        const float v = __half2float(__ldg(hs_row + i));
        thread_sum_sq += v * v;
    }
    thread_sum_sq = warp_all_reduce_sum(thread_sum_sq);

    __shared__ float s_warp_sum[NUM_WARPS];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) s_warp_sum[warp] = thread_sum_sq;
    __syncthreads();
    if (warp == 0) {
        float partial = (lane < NUM_WARPS) ? s_warp_sum[lane] : 0.0f;
        partial = warp_all_reduce_sum(partial);
        if (lane == 0) s_warp_sum[0] = partial;
    }
    __syncthreads();
    const float inv_rms = __frsqrt_rn(
        s_warp_sum[0] / static_cast<float>(HIDDEN_SIZE) + eps);

    // Phase 2: scale + apply norm_weight, write FP16-quantised result
    // directly to the global normalized tensor. The reference's
    // `(norm_weight * hidden_states).to(input_dtype)` step matches
    // this exact precision profile element-wise.
    #pragma unroll
    for (int i = tid; i < HIDDEN_SIZE; i += BLOCK_THREADS) {
        const float v = __half2float(__ldg(hs_row + i)) * inv_rms;
        norm_row[i] = __float2half(
            v * __half2float(__ldg(norm_weight + i)));
    }
}

}  // namespace

void problem_073_rmsnorm_launcher(
    const torch::Tensor& encoder_hidden_states,
    const torch::Tensor& norm_weight,
    torch::Tensor& normalized,
    float eps,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(encoder_hidden_states.size(0));
    const int seq_len = static_cast<int>(encoder_hidden_states.size(1));
    dim3 grid(batch_size * seq_len);
    dim3 block(BLOCK_THREADS);

    problem_073_rmsnorm_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(encoder_hidden_states.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(norm_weight.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(normalized.data_ptr<at::Half>()),
        batch_size,
        seq_len,
        eps);
}
