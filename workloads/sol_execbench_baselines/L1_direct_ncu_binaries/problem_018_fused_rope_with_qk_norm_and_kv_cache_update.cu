// Direct-NCU-harvest binary for the deployed
// problem_018_fused_rope_with_qk_norm_and_kv_cache_update CUDA hand-
// port. The deployed launcher composes 4 kernel launches per call
// (rms_rope_kernel for Q + rms_rope_kernel for K + scatter_k_kernel +
// scatter_v_kernel) plus 2 cudaMemcpyAsync calls. For the NCU
// calibration sweep we need a SINGLE-TARGET probe per
// BL-20260521-ncu-launch-index-single-target-probes — `ncu
// --launch-skip 5 --launch-count 1` only profiles ONE launch
// deterministically when every kernel in the loop body is the same.
// The target kernel under measurement is `rms_rope_kernel` (the RMS
// norm + RoPE pass): its read-only sector traffic is what the warm/
// cold cache-protocol pinning's `measured_cache_initial_state`
// carrier actually controls. The scatter kernels are write-mostly
// helpers; their cache traffic is uniform across warm/cold and is
// not the calibration target.
//
// This binary therefore allocates the inputs `rms_rope_kernel`
// needs (one tensor row + norm weight + position_ids + inv_freq) and
// loops ONLY `rms_rope_kernel` 6 times.
//
// Representative workload: workload index 4 from the canonical
// workload.jsonl (B=1, seq_len=128, cache_len=0), with cohort-A
// shape constants pinned by the problem definition
// (num_kv_heads=8, head_dim=128, half_head_dim=64). We use the K-head
// path (8 heads) as the harvest target — it is the workload-shape
// the warm/cold cache-protocol carrier actually affects (Q goes
// through the same kernel but has 96 heads, 12x the traffic; K is the
// minimal-but-representative shape).
//
// Build via the Makefile in this directory:
//   make ARCH=sm_86
//
// Run under the calibration sweep via
// `scripts/baselines_harness/calibration/cache_prediction/run_ncu_harvest_batch.py`
// with `probe_binary_root` pointed at this directory.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_SAFECALL(call)                                                  \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s in %s:%d: %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(_err));    \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

namespace {

constexpr int kBatch = 1;
constexpr int kNumKvHeads = 8;
constexpr int kSeqLen = 128;
constexpr int kHeadDim = 128;
constexpr int kHalfHeadDim = kHeadDim / 2;
constexpr int kLoopCount = 6;
constexpr float kRmsNormEps = 1e-6f;

// Block-reduce helper — identical to the deployed kernel's.
__device__ __forceinline__ float blockReduceSum_fp32(float val) {
    __shared__ float s_warp[4];

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    if (lane == 0) {
        s_warp[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < 4) ? s_warp[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffffu, val, offset);
        }
    }
    __shared__ float s_total;
    if (threadIdx.x == 0) {
        s_total = val;
    }
    __syncthreads();
    return s_total;
}

// EXACT copy of the deployed kernel from
// baselines/L1/problem_018_fused_rope_with_qk_norm_and_kv_cache_update/
//   cuda_cpp/kernel.cu.
__global__ void rms_rope_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ norm_weight,
    const long* __restrict__ position_ids,
    const float* __restrict__ inv_freq,
    float rms_norm_eps,
    __nv_bfloat16* __restrict__ out,
    int B, int num_heads, int seq_len) {

    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= B || h >= num_heads || s >= seq_len || tid >= kHeadDim) return;

    const long row_base =
        ((static_cast<long>(b) * num_heads + h) * seq_len + s) * kHeadDim;
    const long row_dst_base = row_base;
    const long pos_idx = static_cast<long>(b) * seq_len + s;

    const float x = __bfloat162float(in[row_base + tid]);
    const float sq = x * x;
    const float sum_sq = blockReduceSum_fp32(sq);
    const float mean = sum_sq / static_cast<float>(kHeadDim);
    const float inv_norm = rsqrtf(mean + rms_norm_eps);

    const float w = __bfloat162float(norm_weight[tid]);
    const float normed_f = x * inv_norm * w;
    const __nv_bfloat16 normed_bf = __float2bfloat16(normed_f);

    __shared__ __nv_bfloat16 s_normed[kHeadDim];
    s_normed[tid] = normed_bf;
    __syncthreads();

    const float pos_f = static_cast<float>(position_ids[pos_idx]);
    const int freq_idx = (tid < kHalfHeadDim) ? tid : (tid - kHalfHeadDim);
    const float freq = inv_freq[freq_idx] * pos_f;
    float cf, sf;
    __sincosf(freq, &sf, &cf);
    const __nv_bfloat16 cos_bf = __float2bfloat16(cf);
    const __nv_bfloat16 sin_bf = __float2bfloat16(sf);

    int peer_tid;
    float rotate_sign;
    if (tid < kHalfHeadDim) {
        peer_tid = tid + kHalfHeadDim;
        rotate_sign = -1.0f;
    } else {
        peer_tid = tid - kHalfHeadDim;
        rotate_sign = 1.0f;
    }
    const float peer_f = rotate_sign * __bfloat162float(s_normed[peer_tid]);

    const float own_f = __bfloat162float(normed_bf);
    const float cos_f = __bfloat162float(cos_bf);
    const float sin_f = __bfloat162float(sin_bf);
    const __nv_bfloat16 term1_bf = __float2bfloat16(own_f * cos_f);
    const __nv_bfloat16 term2_bf = __float2bfloat16(peer_f * sin_f);
    const float result =
        __bfloat162float(term1_bf) + __bfloat162float(term2_bf);
    out[row_dst_base + tid] = __float2bfloat16(result);
}

template <typename T>
void* device_alloc(size_t numel) {
    void* p = nullptr;
    CUDA_SAFECALL(cudaMalloc(&p, numel * sizeof(T)));
    CUDA_SAFECALL(cudaMemset(p, 0, numel * sizeof(T)));
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    int loop_count = kLoopCount;
    if (argc > 1) {
        loop_count = atoi(argv[1]);
        if (loop_count < 1) loop_count = kLoopCount;
    }

    const long in_numel =
        static_cast<long>(kBatch) * kNumKvHeads * kSeqLen * kHeadDim;
    const long out_numel = in_numel;
    const long norm_weight_numel = kHeadDim;
    const long position_ids_numel = static_cast<long>(kBatch) * kSeqLen;
    const long inv_freq_numel = kHalfHeadDim;

    __nv_bfloat16* d_in =
        static_cast<__nv_bfloat16*>(device_alloc<__nv_bfloat16>(in_numel));
    __nv_bfloat16* d_norm_weight =
        static_cast<__nv_bfloat16*>(device_alloc<__nv_bfloat16>(norm_weight_numel));
    long* d_position_ids =
        static_cast<long*>(device_alloc<long>(position_ids_numel));
    float* d_inv_freq =
        static_cast<float*>(device_alloc<float>(inv_freq_numel));
    __nv_bfloat16* d_out =
        static_cast<__nv_bfloat16*>(device_alloc<__nv_bfloat16>(out_numel));

    dim3 grid(kSeqLen, kNumKvHeads, kBatch);
    dim3 block(kHeadDim);

    for (int it = 0; it < loop_count; ++it) {
        rms_rope_kernel<<<grid, block>>>(
            d_in, d_norm_weight, d_position_ids, d_inv_freq,
            kRmsNormEps,
            d_out,
            kBatch, kNumKvHeads, kSeqLen);
    }
    CUDA_SAFECALL(cudaDeviceSynchronize());

    CUDA_SAFECALL(cudaFree(d_in));
    CUDA_SAFECALL(cudaFree(d_norm_weight));
    CUDA_SAFECALL(cudaFree(d_position_ids));
    CUDA_SAFECALL(cudaFree(d_inv_freq));
    CUDA_SAFECALL(cudaFree(d_out));
    return 0;
}
