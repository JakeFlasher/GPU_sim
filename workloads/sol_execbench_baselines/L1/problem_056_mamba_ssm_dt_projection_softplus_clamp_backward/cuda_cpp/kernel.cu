#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// BF16 Mamba SSM dt-projection softplus + clamp backward.
//
// Per-head CTA layout (Grid `(num_heads,)`; BLOCK_THREADS = 256):
// each CTA processes ALL `(b, s)` rows for a single `h` via a
// stride-loop. Each thread reads three BF16 elements at offset `h`
// within its row, casts to fp32, applies the clamp_mask + sigmoid
// chain, writes `grad_dt[row, h] = bf16(grad)`, and accumulates a
// fp32 `partial_sum`. After the stride loop the warp-shuffle +
// 8-way cross-warp tree reduces `partial_sum` to a single CTA
// scalar; thread 0 casts to BF16 and writes
// `grad_dt_bias[h] = bf16(total_sum)`. This matches the reference's
// `grad.sum(dim=(0, 1)).to(torch.bfloat16)` (sum in fp32 BEFORE the
// BF16 cast).
//
// Read access pattern: each warp's 32 contiguous threads read
// 32 contiguous (b, s) rows at offset `h` within each row, so the
// stride between adjacent threads' loads is `num_heads = 40`
// elements = 80 bytes — a strided non-coalesced read by design.
// This is the slug's natural memory access shape; the cohort-B
// probe must mirror it exactly.

namespace {

constexpr int kNumHeads = 40;
constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockThreads / kWarpSize;  // 8

__device__ inline float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

__global__ void __launch_bounds__(kBlockThreads, 4)
problem_056_mamba_softplus_clamp_backward_kernel(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ dt_with_bias,
    const __nv_bfloat16* __restrict__ dt_activated,
    __nv_bfloat16* __restrict__ grad_dt,
    __nv_bfloat16* __restrict__ grad_dt_bias,
    int num_rows,
    int num_heads,
    float time_step_min,
    float time_step_max) {

    const int h = blockIdx.x;
    const int tid = threadIdx.x;

    float partial_sum = 0.0f;
    for (int row = tid; row < num_rows; row += kBlockThreads) {
        const int idx = row * num_heads + h;
        const float go = __bfloat162float(grad_output[idx]);
        const float dwb = __bfloat162float(dt_with_bias[idx]);
        const float dac = __bfloat162float(dt_activated[idx]);

        // Clamp mask: pass gradient only if strictly within bounds.
        const float clamp_pass =
            (dac > time_step_min && dac < time_step_max) ? 1.0f : 0.0f;
        // softplus gradient = sigmoid(dt_with_bias).
        const float softplus_grad = 1.0f / (1.0f + __expf(-dwb));
        const float grad = go * clamp_pass * softplus_grad;

        grad_dt[idx] = __float2bfloat16_rn(grad);
        partial_sum += grad;
    }

    // Warp-shuffle + 8-way cross-warp reduction.
    partial_sum = warp_reduce_sum(partial_sum);
    __shared__ float warp_sums[kNumWarps];
    const int warp_id = tid / kWarpSize;
    const int lane_id = tid % kWarpSize;
    if (lane_id == 0) {
        warp_sums[warp_id] = partial_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < kNumWarps; ++w) {
            total += warp_sums[w];
        }
        grad_dt_bias[h] = __float2bfloat16_rn(total);
    }
}

}  // namespace

void problem_056_mamba_softplus_clamp_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& dt_with_bias,
    const torch::Tensor& dt_activated,
    float time_step_min,
    float time_step_max,
    torch::Tensor& grad_dt,
    torch::Tensor& grad_dt_bias,
    cudaStream_t stream) {

    const int64_t batch_size = grad_output.size(0);
    const int64_t seq_len = grad_output.size(1);
    const int64_t num_heads = grad_output.size(2);
    const int num_rows = static_cast<int>(batch_size * seq_len);

    dim3 grid(static_cast<int>(num_heads));
    dim3 block(kBlockThreads);

    problem_056_mamba_softplus_clamp_backward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            grad_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            dt_with_bias.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            dt_activated.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            grad_dt.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            grad_dt_bias.data_ptr<at::BFloat16>()),
        num_rows,
        static_cast<int>(num_heads),
        time_step_min,
        time_step_max);
}
