#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// BF16 tanh-gated residual-addition backward.
//
// Kernel A: per-(b, s) row CTA over the pinned hidden_size = 4096
// BF16 elements. Each thread (1) writes grad_hidden_states[b, s, h]
// = bf16(grad_output[b, s, h].f32 * tanh_g * mask[b, s, 0].f32) and
// (2) accumulates partial_sum += grad_output[b, s, h] *
// hidden_states[b, s, h] * mask[b, s, 0] in fp32. Warp-shuffle +
// 8-way cross-warp tree reduces partial_sum to a single fp32 per
// CTA; thread 0 writes row_partials[b*S + s].
//
// Kernel B: single-CTA reduction of the row_partials[N=B*S] fp32
// buffer. BLOCK_THREADS = 256. Each thread stride-loops over the
// buffer, accumulating into a fp32 register; warp-shuffle + 8-way
// cross-warp tree reduces; thread 0 multiplies by sech² and writes
// bf16(total_sum * sech2) to the zero-dim grad_gate output.

namespace {

constexpr int kHiddenSize = 4096;
constexpr int kHiddenPairs = kHiddenSize / 2;  // BF16x2 view
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
problem_061_grad_hidden_kernel(
    const __nv_bfloat16* __restrict__ grad_output,
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ mask,
    __nv_bfloat16* __restrict__ grad_residual,
    __nv_bfloat16* __restrict__ grad_hidden_states,
    float* __restrict__ row_partials,
    float tanh_g) {

    const int row = blockIdx.x;          // row = b * S + s
    const int tid = threadIdx.x;
    const int row_offset = row * kHiddenSize;
    const __nv_bfloat16 mask_b16 = mask[row];      // mask[b, s, 0]
    const float mask_val = __bfloat162float(mask_b16);
    // Broadcast pair for the bf16-multiply-then-cast contract.
    const __nv_bfloat162 mask_pair_b16 =
        __bfloat162bfloat162(mask_b16);

    const __nv_bfloat162* go_row_vec =
        reinterpret_cast<const __nv_bfloat162*>(grad_output + row_offset);
    const __nv_bfloat162* hs_row_vec =
        reinterpret_cast<const __nv_bfloat162*>(hidden_states + row_offset);
    __nv_bfloat162* ghs_row_vec =
        reinterpret_cast<__nv_bfloat162*>(grad_hidden_states + row_offset);

    float partial_sum = 0.0f;
    const float gate_mask = tanh_g * mask_val;

    for (int i = tid; i < kHiddenPairs; i += kBlockThreads) {
        const __nv_bfloat162 g_pair = __ldg(go_row_vec + i);
        const __nv_bfloat162 h_pair = __ldg(hs_row_vec + i);
        const float2 g_f2 = __bfloat1622float2(g_pair);

        // grad_hidden_states = grad_output * tanh_g * mask  (fp32 math,
        // matching the reference's fp32 cast chain `grad_output * tanh_g *
        // mask` before the final bf16 store).
        const float y0 = g_f2.x * gate_mask;
        const float y1 = g_f2.y * gate_mask;
        reinterpret_cast<__nv_bfloat162*>(grad_residual + row_offset)[i] = g_pair;
        ghs_row_vec[i] = __float22bfloat162_rn(make_float2(y0, y1));

        // Accumulator for grad_gate:
        //   sum(grad_output.f32 * (hidden_states * mask).bf16.f32)
        // The reference computes `masked_hidden_states = hidden_states *
        // mask` in BF16 BEFORE casting to fp32 (both factors are BF16, so
        // PyTorch's default dispatch runs the multiply in BF16 with BF16
        // rounding). To match the reference's numerics exactly, multiply
        // `h_pair * mask_pair_b16` in BF16 via `__hmul2` first, then
        // cast the product to fp32 and accumulate.
        const __nv_bfloat162 hm_pair = __hmul2(h_pair, mask_pair_b16);
        const float2 hm_f2 = __bfloat1622float2(hm_pair);
        partial_sum += g_f2.x * hm_f2.x + g_f2.y * hm_f2.y;
    }

    // Warp + cross-warp reduction.
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
        row_partials[row] = total;
    }
}

__global__ void __launch_bounds__(kBlockThreads, 1)
problem_061_grad_gate_reduce_kernel(
    const float* __restrict__ row_partials,
    __nv_bfloat16* __restrict__ grad_gate,
    int num_rows,
    float sech2) {

    const int tid = threadIdx.x;

    float partial = 0.0f;
    for (int i = tid; i < num_rows; i += kBlockThreads) {
        partial += row_partials[i];
    }

    partial = warp_reduce_sum(partial);
    __shared__ float warp_sums[kNumWarps];
    const int warp_id = tid / kWarpSize;
    const int lane_id = tid % kWarpSize;
    if (lane_id == 0) {
        warp_sums[warp_id] = partial;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < kNumWarps; ++w) {
            total += warp_sums[w];
        }
        const float result = sech2 * total;
        grad_gate[0] = __float2bfloat16_rn(result);
    }
}

}  // namespace

void problem_061_grad_hidden_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& mask,
    float tanh_g,
    torch::Tensor& grad_residual,
    torch::Tensor& grad_hidden_states,
    torch::Tensor& row_partials,
    cudaStream_t stream) {

    const int64_t batch_size = grad_output.size(0);
    const int64_t seq_len = grad_output.size(1);
    const int num_rows = static_cast<int>(batch_size * seq_len);

    dim3 grid(num_rows);
    dim3 block(kBlockThreads);

    problem_061_grad_hidden_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            grad_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            hidden_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            mask.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            grad_residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            grad_hidden_states.data_ptr<at::BFloat16>()),
        row_partials.data_ptr<float>(),
        tanh_g);
}

void problem_061_grad_gate_reduce_launcher(
    const torch::Tensor& row_partials,
    float sech2,
    torch::Tensor& grad_gate,
    cudaStream_t stream) {

    const int num_rows = static_cast<int>(row_partials.numel());

    dim3 grid(1);
    dim3 block(kBlockThreads);

    problem_061_grad_gate_reduce_kernel<<<grid, block, 0, stream>>>(
        row_partials.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(
            grad_gate.data_ptr<at::BFloat16>()),
        num_rows,
        sech2);
}
