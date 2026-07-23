#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Two-kernel decomposition of the RMSNorm backward.
//
//   row_sum[b, s]      = sum_h grad_output_f32[b, s, h] * weight[h] * normalized[b, s, h]
//   mean_grad_norm     = row_sum / H
//   grad_x[b, s, h]    = rstd[b, s] * (grad_output_f32[b, s, h] * weight[h]
//                                       - mean_grad_norm[b, s] * normalized[b, s, h])
//   grad_weight[h]     = sum_{b, s} grad_output_f32[b, s, h] * normalized[b, s, h]
//
// grad_hidden_states and grad_residual receive the same bf16 cast of
// grad_x (the reference does `grad_residual = grad_x_bf16.clone()`).
//
// Kernel A: one CTA per (b, s) row; 256 threads stride along H=2560
// in contiguous loads. Two passes over the row — first to accumulate
// row_sum, second to write grad_x.
//
// Kernel B: one CTA per h column. 256 threads grid-stride across the
// num_rows = B*S elements at fixed h. Strided global reads — slow on
// the local 3090 Ti but correct; this is the predictor-hostile cohort
// for the cache-prediction calibration. A future B200 tuning pass can
// replace this with a TMA-driven reduction over a workspace.

namespace {

constexpr int HIDDEN_SIZE = 2560;
constexpr int BLOCK_THREADS = 256;
constexpr int MAX_WARPS = BLOCK_THREADS / 32;  // 8

__device__ inline float warp_all_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

__device__ inline float block_reduce_sum(float val, float* s_warp) {
    val = warp_all_reduce_sum(val);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) s_warp[warp] = val;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < MAX_WARPS) ? s_warp[lane] : 0.0f;
        v = warp_all_reduce_sum(v);
        if (lane == 0) s_warp[0] = v;
    }
    __syncthreads();
    return s_warp[0];
}

__global__ void __launch_bounds__(BLOCK_THREADS, 4)
grad_x_kernel(
    const __nv_bfloat16* __restrict__ grad_output,
    const float* __restrict__ normalized,
    const float* __restrict__ rstd,
    const float* __restrict__ weight,
    __nv_bfloat16* __restrict__ grad_hidden_states,
    __nv_bfloat16* __restrict__ grad_residual,
    int num_rows) {

    const int row = blockIdx.x;
    if (row >= num_rows) return;

    const long row_off = static_cast<long>(row) * HIDDEN_SIZE;
    const __nv_bfloat16* go_row = grad_output + row_off;
    const float* nrm_row = normalized + row_off;
    __nv_bfloat16* ghs_row = grad_hidden_states + row_off;
    __nv_bfloat16* gres_row = grad_residual + row_off;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_THREADS) {
        float go = __bfloat162float(__ldg(go_row + i));
        float w = __ldg(weight + i);
        float nrm = __ldg(nrm_row + i);
        local_sum += go * w * nrm;
    }

    __shared__ float s_warp[MAX_WARPS];
    const float row_sum = block_reduce_sum(local_sum, s_warp);
    const float mean_grad_norm = row_sum / static_cast<float>(HIDDEN_SIZE);
    const float rstd_row = __ldg(rstd + row);

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_THREADS) {
        float go = __bfloat162float(__ldg(go_row + i));
        float w = __ldg(weight + i);
        float nrm = __ldg(nrm_row + i);
        float grad_norm = go * w;
        float grad_x_val = rstd_row * (grad_norm - mean_grad_norm * nrm);
        __nv_bfloat16 v = __float2bfloat16(grad_x_val);
        ghs_row[i] = v;
        gres_row[i] = v;
    }
}

__global__ void __launch_bounds__(BLOCK_THREADS, 4)
grad_weight_kernel(
    const __nv_bfloat16* __restrict__ grad_output,
    const float* __restrict__ normalized,
    float* __restrict__ grad_weight,
    int num_rows) {

    const int h = blockIdx.x;
    if (h >= HIDDEN_SIZE) return;

    float local_sum = 0.0f;
    for (int row = threadIdx.x; row < num_rows; row += BLOCK_THREADS) {
        long off = static_cast<long>(row) * HIDDEN_SIZE + h;
        float go = __bfloat162float(__ldg(grad_output + off));
        float nrm = __ldg(normalized + off);
        local_sum += go * nrm;
    }

    __shared__ float s_warp[MAX_WARPS];
    const float total = block_reduce_sum(local_sum, s_warp);
    if (threadIdx.x == 0) grad_weight[h] = total;
}

}  // namespace

void problem_013_fused_residual_rms_norm_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& /*x*/,
    const torch::Tensor& normalized,
    const torch::Tensor& rstd,
    const torch::Tensor& weight,
    torch::Tensor& grad_hidden_states,
    torch::Tensor& grad_residual,
    torch::Tensor& grad_weight,
    cudaStream_t stream) {

    const int B = static_cast<int>(grad_output.size(0));
    const int S = static_cast<int>(grad_output.size(1));
    const long num_rows = static_cast<long>(B) * static_cast<long>(S);

    const auto* go_ptr = reinterpret_cast<const __nv_bfloat16*>(
        grad_output.data_ptr<at::BFloat16>());
    const float* nrm_ptr = normalized.data_ptr<float>();
    const float* rstd_ptr = rstd.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    auto* ghs_ptr = reinterpret_cast<__nv_bfloat16*>(
        grad_hidden_states.data_ptr<at::BFloat16>());
    auto* gres_ptr = reinterpret_cast<__nv_bfloat16*>(
        grad_residual.data_ptr<at::BFloat16>());
    float* gw_ptr = grad_weight.data_ptr<float>();

    {
        dim3 grid(static_cast<int>(num_rows));
        dim3 block(BLOCK_THREADS);
        grad_x_kernel<<<grid, block, 0, stream>>>(
            go_ptr, nrm_ptr, rstd_ptr, w_ptr,
            ghs_ptr, gres_ptr, static_cast<int>(num_rows));
    }
    {
        dim3 grid(HIDDEN_SIZE);
        dim3 block(BLOCK_THREADS);
        grad_weight_kernel<<<grid, block, 0, stream>>>(
            go_ptr, nrm_ptr, gw_ptr, static_cast<int>(num_rows));
    }
}
