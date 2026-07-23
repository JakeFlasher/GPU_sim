#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Two custom kernels for this fused pattern:
//
//   1) fused_residual_rms_kernel — per-row pipeline:
//        x       = residual + hidden_states                (bf16 → fp32)
//        rstd    = rsqrt(mean(x^2) + eps)
//        norm_out = (x * rstd) * norm_weight               (back to bf16)
//      One CTA per (b, s) row. 256 threads stride along hidden_size in
//      bf16x2 vector pairs. Two passes over the row — first to
//      accumulate sum-of-squares, second to compute and write the
//      bf16 output.
//
//   2) swiglu_kernel — elementwise:
//        intermediate = silu(gate_out) * up_out             (in-place
//        permitted; the launcher writes back into `gate_out` storage).
//      Elementwise grid-stride, bf16x2 vectorized.
//
// The three projections (gate, up, down) are dispatched through
// at::matmul_out on the current CUDA stream — cuBLAS handles are
// owned and stream-bound by PyTorch (CLAUDE.md Rule 5: cuBLAS is a
// host API). The launcher composes the pipeline.

namespace {

constexpr int HIDDEN_SIZE = 16384;
constexpr int HIDDEN_SIZE_VEC = HIDDEN_SIZE / 2;        // bf16x2 pairs per row
constexpr int BLOCK_THREADS = 256;
constexpr int MAX_WARPS = BLOCK_THREADS / 32;           // 8

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
fused_residual_rms_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ norm_out,
    float eps,
    int num_rows) {

    const int row = blockIdx.x;
    if (row >= num_rows) return;

    const long row_off = static_cast<long>(row) * HIDDEN_SIZE;
    const __nv_bfloat162* hs_row =
        reinterpret_cast<const __nv_bfloat162*>(hidden_states + row_off);
    const __nv_bfloat162* res_row =
        reinterpret_cast<const __nv_bfloat162*>(residual + row_off);
    const __nv_bfloat162* w_vec =
        reinterpret_cast<const __nv_bfloat162*>(norm_weight);
    __nv_bfloat162* out_row =
        reinterpret_cast<__nv_bfloat162*>(norm_out + row_off);

    // The reference adds `residual + hidden_states` in bf16 BEFORE
    // casting to fp32, so we must match that rounding to land within
    // tolerance on bf16 outputs. `__hadd2` is the bf16x2 add.
    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE_VEC; i += BLOCK_THREADS) {
        __nv_bfloat162 hv = __ldg(hs_row + i);
        __nv_bfloat162 rv = __ldg(res_row + i);
        __nv_bfloat162 sum_bf = __hadd2(hv, rv);
        float2 x_f2 = __bfloat1622float2(sum_bf);
        thread_sum_sq += x_f2.x * x_f2.x + x_f2.y * x_f2.y;
    }

    __shared__ float s_warp[MAX_WARPS];
    const float row_sum_sq = block_reduce_sum(thread_sum_sq, s_warp);
    const float inv_rms =
        rsqrtf(row_sum_sq / static_cast<float>(HIDDEN_SIZE) + eps);

    for (int i = threadIdx.x; i < HIDDEN_SIZE_VEC; i += BLOCK_THREADS) {
        __nv_bfloat162 hv = __ldg(hs_row + i);
        __nv_bfloat162 rv = __ldg(res_row + i);
        __nv_bfloat162 wv = __ldg(w_vec + i);
        __nv_bfloat162 sum_bf = __hadd2(hv, rv);
        float2 x_f2 = __bfloat1622float2(sum_bf);
        float2 w_f2 = __bfloat1622float2(wv);
        float y0 = x_f2.x * inv_rms * w_f2.x;
        float y1 = x_f2.y * inv_rms * w_f2.y;
        out_row[i] = __float22bfloat162_rn(make_float2(y0, y1));
    }
}

__device__ inline float silu_f(float v) {
    return v / (1.0f + __expf(-v));
}

__global__ void __launch_bounds__(BLOCK_THREADS, 4)
swiglu_kernel(
    const __nv_bfloat162* __restrict__ gate_out,
    const __nv_bfloat162* __restrict__ up_out,
    __nv_bfloat162* __restrict__ intermediate,
    long num_pairs) {

    // The reference computes `F.silu(gate_output)` (bf16) and then
    // multiplies by `up_output` (bf16). Both intermediate stores are
    // bf16, so we round the silu output to bf16 before the multiply
    // to match the reference's lower-precision intermediate.
    long stride = static_cast<long>(blockDim.x) * gridDim.x;
    long i_start = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (long i = i_start; i < num_pairs; i += stride) {
        __nv_bfloat162 gv = __ldg(gate_out + i);
        __nv_bfloat162 uv = __ldg(up_out + i);
        float2 g_f2 = __bfloat1622float2(gv);
        float2 u_f2 = __bfloat1622float2(uv);
        __nv_bfloat162 silu_bf = __float22bfloat162_rn(
            make_float2(silu_f(g_f2.x), silu_f(g_f2.y)));
        float2 silu_f2 = __bfloat1622float2(silu_bf);
        intermediate[i] = __float22bfloat162_rn(
            make_float2(silu_f2.x * u_f2.x, silu_f2.y * u_f2.y));
    }
}

}  // namespace

void fused_residual_rms_kernel_launch(
    const at::BFloat16* hidden_states,
    const at::BFloat16* residual,
    const at::BFloat16* norm_weight,
    at::BFloat16* norm_out,
    float eps,
    int num_rows,
    cudaStream_t stream) {

    dim3 grid(num_rows);
    dim3 block(BLOCK_THREADS);
    fused_residual_rms_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states),
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight),
        reinterpret_cast<__nv_bfloat16*>(norm_out),
        eps,
        num_rows);
}

void swiglu_kernel_launch(
    const at::BFloat16* gate_out,
    const at::BFloat16* up_out,
    at::BFloat16* intermediate,
    long num_elems,
    cudaStream_t stream) {

    const long num_pairs = num_elems / 2;
    constexpr int blocks_per_sm = 32;
    constexpr int num_sms = 84;  // 3090 Ti; over-provisioning is harmless
    long num_blocks = std::min<long>(
        static_cast<long>(blocks_per_sm) * num_sms,
        (num_pairs + BLOCK_THREADS - 1) / BLOCK_THREADS);
    if (num_blocks <= 0) num_blocks = 1;
    dim3 grid(static_cast<unsigned int>(num_blocks));
    dim3 block(BLOCK_THREADS);
    swiglu_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(gate_out),
        reinterpret_cast<const __nv_bfloat162*>(up_out),
        reinterpret_cast<__nv_bfloat162*>(intermediate),
        num_pairs);
}

void problem_004_fused_residual_rms_mlp_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    const torch::Tensor& gate_proj_weight,
    const torch::Tensor& up_proj_weight,
    const torch::Tensor& down_proj_weight,
    float eps,
    torch::Tensor& output,
    cudaStream_t stream) {

    const int B = static_cast<int>(hidden_states.size(0));
    const int S = static_cast<int>(hidden_states.size(1));
    const int H = static_cast<int>(hidden_states.size(2));
    const int I = static_cast<int>(gate_proj_weight.size(0));
    const long num_rows = static_cast<long>(B) * static_cast<long>(S);

    auto bf16_options = hidden_states.options();

    // Per-launch intermediates. norm_out: [B, S, H]; gate_out / up_out:
    // [B, S, I]. After SwiGLU the result is written back over gate_out
    // storage so we do not allocate a separate intermediate buffer.
    auto norm_out = torch::empty({B, S, H}, bf16_options);
    auto gate_out = torch::empty({B, S, I}, bf16_options);
    auto up_out = torch::empty({B, S, I}, bf16_options);

    // 1) Fused residual + RMSNorm.
    fused_residual_rms_kernel_launch(
        hidden_states.data_ptr<at::BFloat16>(),
        residual.data_ptr<at::BFloat16>(),
        norm_weight.data_ptr<at::BFloat16>(),
        norm_out.data_ptr<at::BFloat16>(),
        eps,
        static_cast<int>(num_rows),
        stream);

    // 2) Gate / up projections (cuBLAS via at::matmul_out on the
    //    current stream).
    auto gate_weight_t = gate_proj_weight.t();
    auto up_weight_t = up_proj_weight.t();
    at::matmul_out(gate_out, norm_out, gate_weight_t);
    at::matmul_out(up_out, norm_out, up_weight_t);

    // 3) SwiGLU activation, in-place over gate_out storage.
    swiglu_kernel_launch(
        gate_out.data_ptr<at::BFloat16>(),
        up_out.data_ptr<at::BFloat16>(),
        gate_out.data_ptr<at::BFloat16>(),
        static_cast<long>(num_rows) * static_cast<long>(I),
        stream);

    // 4) Down projection into the DPS output tensor.
    auto down_weight_t = down_proj_weight.t();
    at::matmul_out(output, gate_out, down_weight_t);

    (void)stream;  // cuBLAS picks up the current stream internally.
}
