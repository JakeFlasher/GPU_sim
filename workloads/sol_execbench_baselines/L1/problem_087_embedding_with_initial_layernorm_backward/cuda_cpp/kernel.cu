#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Embedding + RMSNorm backward. The PyTorch reference computes:
//   grad_out_fp32 = grad_output.to(fp32)
//   h_norm = hidden_states_fp32 * rstd          (per (b,s,h))
//   grad_norm_weight[h] = sum_(b,s) grad_out_fp32 * h_norm
//   grad_h_norm = grad_out_fp32 * norm_weight.to(fp32)
//   mean = mean_h(grad_h_norm * h_norm)         (per (b,s))
//   grad_h = rstd * (grad_h_norm - mean * h_norm)
//   grad_embed_weight[input_ids[b,s]] += grad_h    (scatter-add)
//
// Custom-kernel design: one CTA per (b,s) row drives the per-row
// reduction + scatter-add. Three FP32 shared arrays cache per-h
// values across the reduction barrier so we do not re-read global
// memory in the writeback phase. Two trivial cast kernels finalize
// the BF16 DPS outputs.

namespace {

constexpr int HIDDEN_SIZE = 4096;
constexpr int VOCAB_SIZE = 65536;
constexpr int BLOCK_THREADS = 256;
constexpr int NUM_WARPS = BLOCK_THREADS / 32;  // 8

__device__ inline float warp_all_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

__global__ void __launch_bounds__(BLOCK_THREADS, 2)
problem_087_embedding_layernorm_backward_kernel(
    const __nv_bfloat16* __restrict__ grad_output,
    const int64_t* __restrict__ input_ids,
    const float* __restrict__ hidden_states_fp32,
    const float* __restrict__ rstd,
    const __nv_bfloat16* __restrict__ norm_weight,
    float* __restrict__ grad_embed_weight_fp32,
    float* __restrict__ grad_norm_weight_fp32,
    int batch_size,
    int seq_len) {

    const int row_idx = blockIdx.x;
    const int total_rows = batch_size * seq_len;
    if (row_idx >= total_rows) return;

    const int tid = threadIdx.x;

    const __nv_bfloat16* grad_out_row =
        grad_output + row_idx * HIDDEN_SIZE;
    const float* hidden_row = hidden_states_fp32 + row_idx * HIDDEN_SIZE;
    const float rstd_bs = rstd[row_idx];
    const int64_t input_id = input_ids[row_idx];

    // Shared caches for the second pass. Storing
    //   h_norm[h]      (FP32)
    //   grad_h_norm[h] (FP32) = grad_out * norm_weight
    // keeps the writeback pass off the FP32 hidden-state global tile.
    // grad_out is re-loaded from BF16 gmem in Phase 3 (cheap; L1
    // hits the just-touched tile) so the kernel stays inside the
    // 48 KiB static-shared limit instead of opting into dynamic
    // shared.
    __shared__ float s_h_norm[HIDDEN_SIZE];
    __shared__ float s_grad_h_norm[HIDDEN_SIZE];

    // Phase 1: load + compute per-(h) values, fill shared, accumulate
    // the per-row mean-reduction partial.
    float thread_partial = 0.0f;
    #pragma unroll
    for (int i = tid; i < HIDDEN_SIZE; i += BLOCK_THREADS) {
        const float grad_out_h =
            __bfloat162float(__ldg(grad_out_row + i));
        const float h_norm_h = __ldg(hidden_row + i) * rstd_bs;
        const float norm_w_h =
            __bfloat162float(__ldg(norm_weight + i));
        const float grad_h_norm_h = grad_out_h * norm_w_h;
        s_h_norm[i] = h_norm_h;
        s_grad_h_norm[i] = grad_h_norm_h;
        thread_partial += grad_h_norm_h * h_norm_h;
    }

    // Phase 2: block-wide sum reduction → mean across HIDDEN_SIZE.
    thread_partial = warp_all_reduce_sum(thread_partial);
    __shared__ float s_warp_sum[NUM_WARPS];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) s_warp_sum[warp] = thread_partial;
    __syncthreads();
    if (warp == 0) {
        float partial = (lane < NUM_WARPS) ? s_warp_sum[lane] : 0.0f;
        partial = warp_all_reduce_sum(partial);
        if (lane == 0) s_warp_sum[0] = partial;
    }
    __syncthreads();
    const float mean =
        s_warp_sum[0] / static_cast<float>(HIDDEN_SIZE);

    // Phase 3: writeback. Per h:
    //   grad_h = rstd * (grad_h_norm - mean * h_norm)
    //   atomicAdd → grad_embed_weight_fp32[input_id, h]
    //   atomicAdd → grad_norm_weight_fp32[h]
    float* grad_embed_row =
        grad_embed_weight_fp32 + input_id * HIDDEN_SIZE;
    #pragma unroll
    for (int i = tid; i < HIDDEN_SIZE; i += BLOCK_THREADS) {
        const float h_norm_h = s_h_norm[i];
        const float grad_h_norm_h = s_grad_h_norm[i];
        const float grad_h_fp32 =
            rstd_bs * (grad_h_norm_h - mean * h_norm_h);
        atomicAdd(grad_embed_row + i, grad_h_fp32);
        const float grad_out_h =
            __bfloat162float(__ldg(grad_out_row + i));
        atomicAdd(grad_norm_weight_fp32 + i, grad_out_h * h_norm_h);
    }
}

__global__ void cast_fp32_to_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t numel) {

    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = __float2bfloat16(src[idx]);
}

}  // namespace

void problem_087_embedding_layernorm_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& input_ids,
    const torch::Tensor& hidden_states_fp32,
    const torch::Tensor& rstd,
    const torch::Tensor& norm_weight,
    torch::Tensor& grad_embed_weight,
    torch::Tensor& grad_norm_weight,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(grad_output.size(0));
    const int seq_len = static_cast<int>(grad_output.size(1));
    const int64_t total_rows =
        static_cast<int64_t>(batch_size) * seq_len;

    // FP32 accumulators. zeros() initializes via cudaMemsetAsync on
    // the current stream so the kernel sees zeroed memory.
    auto options_fp32 = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(grad_output.device());
    torch::Tensor grad_embed_weight_fp32 =
        torch::zeros({VOCAB_SIZE, HIDDEN_SIZE}, options_fp32);
    torch::Tensor grad_norm_weight_fp32 =
        torch::zeros({HIDDEN_SIZE}, options_fp32);

    dim3 grid(static_cast<unsigned int>(total_rows));
    dim3 block(BLOCK_THREADS);

    problem_087_embedding_layernorm_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            grad_output.data_ptr<at::BFloat16>()),
        input_ids.data_ptr<int64_t>(),
        hidden_states_fp32.data_ptr<float>(),
        rstd.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(
            norm_weight.data_ptr<at::BFloat16>()),
        grad_embed_weight_fp32.data_ptr<float>(),
        grad_norm_weight_fp32.data_ptr<float>(),
        batch_size,
        seq_len);

    // Cast accumulators → BF16 DPS outputs.
    constexpr int CAST_BLOCK = 256;
    const int64_t embed_numel =
        static_cast<int64_t>(VOCAB_SIZE) * HIDDEN_SIZE;
    const int64_t embed_grid =
        (embed_numel + CAST_BLOCK - 1) / CAST_BLOCK;
    cast_fp32_to_bf16_kernel<<<
        static_cast<unsigned int>(embed_grid),
        CAST_BLOCK, 0, stream>>>(
        grad_embed_weight_fp32.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(
            grad_embed_weight.data_ptr<at::BFloat16>()),
        embed_numel);

    const int64_t norm_numel = HIDDEN_SIZE;
    const int64_t norm_grid =
        (norm_numel + CAST_BLOCK - 1) / CAST_BLOCK;
    cast_fp32_to_bf16_kernel<<<
        static_cast<unsigned int>(norm_grid),
        CAST_BLOCK, 0, stream>>>(
        grad_norm_weight_fp32.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(
            grad_norm_weight.data_ptr<at::BFloat16>()),
        norm_numel);
}
