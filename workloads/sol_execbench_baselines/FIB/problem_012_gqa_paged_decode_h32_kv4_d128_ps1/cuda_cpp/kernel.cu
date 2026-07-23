// GQA paged decode kernel: matches the Python reference
// at references/NVIDIA_Manuals/solbench_problems/problem_012_gqa_paged_decode_h32_kv4_d128_ps1.md.
//
// Parallelization: one block per (batch, query_head)
// pair; 128 threads per block (one per head_dim
// element). Online softmax (Flash-Attention style)
// keeps the per-block memory small.
//
// Precision: FP32 accumulators throughout; BF16 inputs
// read into float; BF16 cast at the output write.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "kernel.h"

namespace {

constexpr int HEAD_DIM = 128;
constexpr int NUM_QO_HEADS = 32;
constexpr int NUM_KV_HEADS = 4;
constexpr int GQA_RATIO = NUM_QO_HEADS / NUM_KV_HEADS;  // 8

// Block-wide reduction across 128 threads = 4 warps. The
// warp shuffle reduces within each warp; the per-warp
// partials are written to shared memory and reduced by
// warp 0.
__device__ __forceinline__ float block_sum_128(
    float v, float* sdata_4) {
    // Intra-warp reduction.
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) sdata_4[warp] = v;
    __syncthreads();
    float r = 0.0f;
    if (warp == 0) {
        float w = (lane < 4) ? sdata_4[lane] : 0.0f;
        for (int offset = 2; offset > 0; offset >>= 1) {
            w += __shfl_xor_sync(0xffffffff, w, offset);
        }
        if (lane == 0) sdata_4[0] = w;
    }
    __syncthreads();
    return sdata_4[0];
}

__global__ void gqa_paged_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    float sm_scale,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    int batch_size) {

    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    if (b >= batch_size || h >= NUM_QO_HEADS) return;

    const int kv_head = h / GQA_RATIO;

    const int page_start = kv_indptr[b];
    const int page_end = kv_indptr[b + 1];
    const int num_tokens = page_end - page_start;

    const int output_offset = (b * NUM_QO_HEADS + h) * HEAD_DIM;

    if (num_tokens <= 0) {
        if (tid < HEAD_DIM) {
            output[output_offset + tid] = __float2bfloat16(0.0f);
        }
        if (tid == 0) {
            lse[b * NUM_QO_HEADS + h] = -INFINITY;
        }
        return;
    }

    // Load q[b, h, tid] once into a register.
    const float q_val = (tid < HEAD_DIM)
        ? __bfloat162float(q[output_offset + tid])
        : 0.0f;

    // Shared scratch:
    //   sdata_dot[0..3] -- 4-warp reduction buffer for dot products
    //   m_smem        -- running max
    //   l_smem        -- running sum_exp
    //   logit_smem    -- per-token logit (broadcast to all threads)
    //   rescale_smem  -- old-softmax rescale, published by thread 0
    //   exp_logit_smem -- new-token weight, published by thread 0
    __shared__ float sdata_dot[4];
    __shared__ float m_smem;
    __shared__ float l_smem;
    __shared__ float logit_smem;
    __shared__ float rescale_smem;
    __shared__ float exp_logit_smem;

    if (tid == 0) {
        m_smem = -INFINITY;
        l_smem = 0.0f;
    }
    __syncthreads();

    float acc = 0.0f;

    for (int i = 0; i < num_tokens; ++i) {
        const int token_idx = kv_indices[page_start + i];
        const int kv_offset = (token_idx * NUM_KV_HEADS + kv_head) * HEAD_DIM + tid;
        const float k_val = (tid < HEAD_DIM)
            ? __bfloat162float(k_cache[kv_offset]) : 0.0f;
        const float v_val = (tid < HEAD_DIM)
            ? __bfloat162float(v_cache[kv_offset]) : 0.0f;

        const float prod = q_val * k_val;
        const float dot = block_sum_128(prod, sdata_dot);

        if (tid == 0) {
            logit_smem = dot * sm_scale;
        }
        __syncthreads();
        const float logit = logit_smem;

        // Thread 0 owns the running softmax state and publishes both factors
        // before any output lane consumes them.  Letting every warp read
        // m_smem while thread 0 overwrites it is a read/write race: later
        // warps can observe m_new as m_prev and skip the required rescale.
        if (tid == 0) {
            const float m_prev = m_smem;
            const float m_new = fmaxf(m_prev, logit);
            rescale_smem = (m_prev == -INFINITY) ? 0.0f
                : expf(m_prev - m_new);
            exp_logit_smem = expf(logit - m_new);
            l_smem = l_smem * rescale_smem + exp_logit_smem;
            m_smem = m_new;
        }
        __syncthreads();

        acc = acc * rescale_smem + exp_logit_smem * v_val;
    }

    const float m_final = m_smem;
    const float l_final = l_smem;

    if (tid < HEAD_DIM) {
        const float out_val = acc / l_final;
        output[output_offset + tid] = __float2bfloat16(out_val);
    }
    if (tid == 0) {
        const float lse_val = (m_final + logf(l_final)) * (1.0f / 0.69314718055994530942f);
        // (1/log(2)) = log2(e); m + log(l) is natural log-sum-exp.
        lse[b * NUM_QO_HEADS + h] = lse_val;
    }
}

}  // namespace

void problem_012_gqa_paged_decode_launcher(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& kv_indptr,
    const torch::Tensor& kv_indices,
    float sm_scale,
    torch::Tensor output,
    torch::Tensor lse,
    cudaStream_t stream) {

    const int batch_size = static_cast<int>(q.size(0));

    dim3 grid(batch_size, NUM_QO_HEADS, 1);
    dim3 block(128, 1, 1);

    gqa_paged_decode_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr<at::BFloat16>()),
        kv_indptr.data_ptr<int>(),
        kv_indices.data_ptr<int>(),
        sm_scale,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        batch_size);
}
