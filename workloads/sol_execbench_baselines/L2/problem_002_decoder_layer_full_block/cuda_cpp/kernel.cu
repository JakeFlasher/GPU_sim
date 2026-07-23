#include "kernel.h"
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Two custom kernels paired with at:: ops:
//   - rms_norm_kernel: per-row RMSNorm in fp32. One CTA per row,
//     256 threads grid-strided across hidden_size. Two-pass: sum-of-
//     squares, then write `x * rsqrt(var + eps) * weight`. Used for
//     both the input and post-attention layernorms.
//   - silu_mul_kernel: elementwise `silu(gate) * up` in fp32; the
//     SwiGLU activation post-fusion.
//
// Everything else — Q/K/V/O/gate/up/down projections, RoPE rotation,
// GQA repeat, attention matmul, softmax, residual adds, contiguous
// reshapes — is dispatched through at:: ops, which resolve to cuBLAS
// (host API, CLAUDE.md Rule 5) and PyTorch's own CUDA kernels on the
// current stream.
//
// Defensive `.contiguous()` is forced after each transpose / unsqueeze
// / RoPE-rotation step. Background: on the local 3090 Ti, the kernel
// reliably tripped `cudaErrorLaunchFailure` on `(B=64, S=128)` while
// passing on every smaller workload. Root-cause analysis pointed at
// a non-contiguous-stride interaction with aten's broadcasted multiply
// on the RoPE path. Forcing contiguous layouts after each shape-shift
// op keeps the downstream matmuls on the well-trodden contiguous path.
//
// Compile with `-DDECODER_DEBUG_SYNC` to insert per-stage synchronise
// + error-check fences for bisection; release builds omit them.

#ifdef DECODER_DEBUG_SYNC
#include <cstdio>
#define DECODER_SYNC_STAGE(label) do { \
    cudaError_t _e_sync = cudaDeviceSynchronize(); \
    cudaError_t _e_last = cudaGetLastError(); \
    if (_e_sync != cudaSuccess || _e_last != cudaSuccess) { \
        std::fprintf(stderr, \
            "DECODER_DEBUG_SYNC fault at %s: sync=%s last=%s\n", \
            (label), cudaGetErrorString(_e_sync), \
            cudaGetErrorString(_e_last)); \
    } \
} while (0)
#else
#define DECODER_SYNC_STAGE(label) ((void)0)
#endif

namespace {

constexpr int BLOCK_THREADS = 256;
constexpr int MAX_WARPS = BLOCK_THREADS / 32;

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
rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    float eps,
    int num_rows,
    int hidden_size) {

    const int row = blockIdx.x;
    if (row >= num_rows) return;

    const long row_off = static_cast<long>(row) * hidden_size;
    const float* x_row = x + row_off;
    float* out_row = out + row_off;

    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_THREADS) {
        float v = __ldg(x_row + i);
        thread_sum_sq += v * v;
    }

    __shared__ float s_warp[MAX_WARPS];
    const float sum_sq = block_reduce_sum(thread_sum_sq, s_warp);
    const float inv_rms = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);

    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_THREADS) {
        float v = __ldg(x_row + i);
        float w = __ldg(weight + i);
        out_row[i] = v * inv_rms * w;
    }
}

__global__ void __launch_bounds__(BLOCK_THREADS, 8)
silu_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    long num_elems) {

    long stride = static_cast<long>(blockDim.x) * gridDim.x;
    long i_start = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (long i = i_start; i < num_elems; i += stride) {
        float g = __ldg(gate + i);
        float u = __ldg(up + i);
        float silu = g / (1.0f + __expf(-g));
        out[i] = silu * u;
    }
}

}  // namespace

void rms_norm_kernel_launch(
    const float* x,
    const float* weight,
    float* out,
    float eps,
    int num_rows,
    int hidden_size,
    cudaStream_t stream) {

    dim3 grid(num_rows);
    dim3 block(BLOCK_THREADS);
    rms_norm_kernel<<<grid, block, 0, stream>>>(
        x, weight, out, eps, num_rows, hidden_size);
}

void silu_mul_kernel_launch(
    const float* gate,
    const float* up,
    float* out,
    long num_elems,
    cudaStream_t stream) {

    constexpr int blocks_per_sm = 32;
    constexpr int num_sms = 84;
    long num_blocks = std::min<long>(
        static_cast<long>(blocks_per_sm) * num_sms,
        (num_elems + BLOCK_THREADS - 1) / BLOCK_THREADS);
    if (num_blocks <= 0) num_blocks = 1;
    dim3 grid(static_cast<unsigned int>(num_blocks));
    dim3 block(BLOCK_THREADS);
    silu_mul_kernel<<<grid, block, 0, stream>>>(
        gate, up, out, num_elems);
}

void problem_002_decoder_layer_full_block_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& attention_mask,
    const torch::Tensor& input_layernorm_weight,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& post_attention_layernorm_weight,
    const torch::Tensor& gate_proj_weight,
    const torch::Tensor& up_proj_weight,
    const torch::Tensor& down_proj_weight,
    float rms_norm_eps,
    torch::Tensor& output,
    cudaStream_t stream) {

    constexpr int num_attention_heads = 32;
    constexpr int num_key_value_heads = 8;
    constexpr int head_dim = 128;
    constexpr int num_key_value_groups =
        num_attention_heads / num_key_value_heads;
    const float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int B = static_cast<int>(hidden_states.size(0));
    const int S = static_cast<int>(hidden_states.size(1));
    const int H = static_cast<int>(hidden_states.size(2));
    const int num_rows = B * S;

    auto residual = hidden_states;  // saved for the attention residual

    // === ATTENTION BLOCK ===
    // 1. Input RMSNorm.
    auto x_norm = torch::empty_like(hidden_states);
    rms_norm_kernel_launch(
        hidden_states.data_ptr<float>(),
        input_layernorm_weight.data_ptr<float>(),
        x_norm.data_ptr<float>(),
        rms_norm_eps,
        num_rows, H,
        stream);
    DECODER_SYNC_STAGE("rms_norm_input");

    // 2. QKV projections via cuBLAS.
    auto q = at::matmul(x_norm, q_proj_weight.t());  // [B, S, qkv_head_dim]
    auto k = at::matmul(x_norm, k_proj_weight.t());  // [B, S, kv_head_dim]
    auto v = at::matmul(x_norm, v_proj_weight.t());  // [B, S, kv_head_dim]
    DECODER_SYNC_STAGE("qkv_proj");

    // Reshape into [B, num_heads, S, head_dim]. The .transpose() leaves
    // the result non-contiguous. Force `.contiguous()` immediately so
    // every downstream op (RoPE elementwise, attention matmul) sees a
    // standard memory layout — without this guard the (B=64, S=128)
    // shape reliably trips `cudaErrorLaunchFailure` on the local 3090
    // Ti, presumably from a non-contiguous-stride edge case in an
    // aten broadcast / matmul kernel selection.
    q = q.view({B, S, num_attention_heads, head_dim}).transpose(1, 2).contiguous();
    k = k.view({B, S, num_key_value_heads, head_dim}).transpose(1, 2).contiguous();
    v = v.view({B, S, num_key_value_heads, head_dim}).transpose(1, 2).contiguous();
    DECODER_SYNC_STAGE("qkv_reshape");

    // 3. RoPE. Stage cos / sin into contiguous broadcast tensors and
    //    force the rotation result contiguous before it flows into the
    //    attention matmul.
    auto cos_expanded = cos.unsqueeze(1).contiguous();  // [B, 1, S, head_dim]
    auto sin_expanded = sin.unsqueeze(1).contiguous();

    auto rope = [&](torch::Tensor t) -> torch::Tensor {
        auto t1 = t.slice(/*dim=*/-1, /*start=*/0, /*end=*/head_dim / 2);
        auto t2 = t.slice(/*dim=*/-1, /*start=*/head_dim / 2, /*end=*/head_dim);
        auto rotated = at::cat({-t2, t1}, /*dim=*/-1);
        return (t * cos_expanded + rotated * sin_expanded).contiguous();
    };
    q = rope(q);
    DECODER_SYNC_STAGE("rope_q");
    k = rope(k);
    DECODER_SYNC_STAGE("rope_k");

    // 4. KV repeat for GQA — expand+reshape on the num_kv_heads axis
    //    to broadcast into num_attention_heads. Mirrors the reference's
    //    `key_states[:, :, None, :, :].expand(...).reshape(...)`. The
    //    `.contiguous()` after the reshape forces a fresh allocation
    //    even when `reshape()` could legally return a view, so the
    //    downstream batched matmul sees standard strides.
    k = k.unsqueeze(2).expand({B, num_key_value_heads, num_key_value_groups, S, head_dim})
         .reshape({B, num_attention_heads, S, head_dim}).contiguous();
    v = v.unsqueeze(2).expand({B, num_key_value_heads, num_key_value_groups, S, head_dim})
         .reshape({B, num_attention_heads, S, head_dim}).contiguous();
    DECODER_SYNC_STAGE("gqa_repeat");

    // 5. Attention: softmax(QK^T * scaling + mask) @ V. Force the
    //    K-transpose contiguous: aten's batched matmul backend is
    //    happier with two contiguous inputs.
    auto k_t = k.transpose(2, 3).contiguous();
    auto attn_weights = at::matmul(q, k_t) * scaling;
    DECODER_SYNC_STAGE("attn_qk");
    attn_weights = attn_weights + attention_mask;
    attn_weights = at::softmax(attn_weights, /*dim=*/-1, /*dtype=*/at::kFloat);
    DECODER_SYNC_STAGE("attn_softmax");

    auto attn_output = at::matmul(attn_weights, v);  // [B, H_q, S, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()
        .view({B, S, num_attention_heads * head_dim});
    DECODER_SYNC_STAGE("attn_av_merge");

    // 6. Output projection + residual.
    attn_output = at::matmul(attn_output, o_proj_weight.t());
    auto post_attn = (residual + attn_output).contiguous();
    DECODER_SYNC_STAGE("o_proj_residual");

    // === MLP BLOCK ===
    auto residual2 = post_attn;

    // 7. Post-attention RMSNorm into the same buffer shape.
    auto x_norm2 = torch::empty_like(post_attn);
    rms_norm_kernel_launch(
        post_attn.data_ptr<float>(),
        post_attention_layernorm_weight.data_ptr<float>(),
        x_norm2.data_ptr<float>(),
        rms_norm_eps,
        num_rows, H,
        stream);
    DECODER_SYNC_STAGE("rms_norm_post");

    // 8. SwiGLU MLP. Gate / Up projections then fused silu * up,
    //    then down projection.
    auto gate_out = at::matmul(x_norm2, gate_proj_weight.t());
    auto up_out = at::matmul(x_norm2, up_proj_weight.t());
    DECODER_SYNC_STAGE("mlp_gate_up_proj");
    auto intermediate = torch::empty_like(gate_out);
    silu_mul_kernel_launch(
        gate_out.data_ptr<float>(),
        up_out.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        static_cast<long>(gate_out.numel()),
        stream);
    DECODER_SYNC_STAGE("silu_mul");
    auto mlp_output = at::matmul(intermediate, down_proj_weight.t());
    DECODER_SYNC_STAGE("mlp_down_proj");

    // 9. MLP residual into the DPS output tensor.
    auto final_sum = residual2 + mlp_output;
    output.copy_(final_sum);
    DECODER_SYNC_STAGE("final_copy");

    (void)stream;
}
