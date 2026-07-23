#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// One CTA per row of [batch, hidden_size=1536]. Threads cooperatively
// reduce sum-of-squares of hidden_states in FP32, then rescale
// hidden_states * inv_rms * weight and store BF16. No residual add —
// this is the plain RMSNorm variant captured from DeepSeek-V3/R1 at
// h=1536. Same template as problem_024 (h=2048) with a stride-loop
// over BLOCK_THREADS BF16x2 pairs.

namespace {

__device__ inline float warp_all_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

constexpr int HIDDEN_SIZE = 1536;
constexpr int HIDDEN_SIZE_VEC = 768;   // BF16x2 pairs per row (1536/2)
constexpr int BLOCK_THREADS = 256;     // 3 BF16x2 pairs per thread
constexpr int MAX_WARPS = 32;
constexpr float EPS = 1e-6f;           // pinned by problem definition

__global__ void __launch_bounds__(BLOCK_THREADS, 4)
problem_023_rmsnorm_h1536_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int batch_size) {

    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const __nv_bfloat162* hs_row =
        reinterpret_cast<const __nv_bfloat162*>(hidden_states)
        + row * HIDDEN_SIZE_VEC;
    const __nv_bfloat162* w_vec =
        reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* out_row =
        reinterpret_cast<__nv_bfloat162*>(output) + row * HIDDEN_SIZE_VEC;

    const int tid = threadIdx.x;

    float thread_sum_sq = 0.0f;
    #pragma unroll 1
    for (int i = tid; i < HIDDEN_SIZE_VEC; i += BLOCK_THREADS) {
        __nv_bfloat162 hv = __ldg(hs_row + i);
        float2 h_f2 = __bfloat1622float2(hv);
        thread_sum_sq += h_f2.x * h_f2.x + h_f2.y * h_f2.y;
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

    const float inv_rms = rsqrtf(row_sum / static_cast<float>(HIDDEN_SIZE) + EPS);

    #pragma unroll 1
    for (int i = tid; i < HIDDEN_SIZE_VEC; i += BLOCK_THREADS) {
        __nv_bfloat162 hv = __ldg(hs_row + i);
        __nv_bfloat162 wv = __ldg(w_vec + i);
        float2 h_f2 = __bfloat1622float2(hv);
        float2 w_f2 = __bfloat1622float2(wv);
        float y0 = h_f2.x * inv_rms * w_f2.x;
        float y1 = h_f2.y * inv_rms * w_f2.y;
        out_row[i] = __float22bfloat162_rn(make_float2(y0, y1));
    }
}

}  // namespace

void problem_023_rmsnorm_h1536_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    torch::Tensor& output,
    cudaStream_t stream) {

    const auto batch_size = static_cast<int>(hidden_states.size(0));
    dim3 grid(batch_size);
    dim3 block(BLOCK_THREADS);

    problem_023_rmsnorm_h1536_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        batch_size);
}
