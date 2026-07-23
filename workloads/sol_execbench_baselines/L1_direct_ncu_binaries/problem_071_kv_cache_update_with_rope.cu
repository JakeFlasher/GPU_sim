// Direct-NCU-harvest binary for the deployed
// problem_071_kv_cache_update_with_rope CUDA hand-port. The same
// __global__ kernel as
// baselines/L1/problem_071_kv_cache_update_with_rope/cuda_cpp/kernel.cu
// is compiled here as a standalone calibration executable. No torch
// dependency: raw cudaMalloc + cudaMemset for the inputs, raw kernel
// launch in a 6-iteration loop so `ncu --launch-skip 5 --launch-count
// 1` deterministically lands on the 6th invocation
// (BL-20260521-ncu-launch-index-single-target-probes).
//
// Representative workload: workload index 1 from the canonical
// workload.jsonl — batch_size=2, new_seq_len=512, current_seq_len=0,
// num_kv_heads=10, head_dim=128. cache_len=0 keeps the copy phase
// dormant; the kernel exercises the RoPE-on-new-K + raw-V-copy code
// path only, which is the cache-traffic-of-interest for the warm/cold
// cache-protocol pinning. The 6-iteration loop matches the project's
// existing cohort-B / calibration-warm-cold convention.
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

constexpr int kBatch = 2;
constexpr int kNumKvHeads = 10;
constexpr int kNewSeqLen = 512;
constexpr int kCurrentSeqLen = 0;
constexpr int kHeadDim = 128;
constexpr int kHalfHeadDim = kHeadDim / 2;
constexpr int kLoopCount = 6;

constexpr int kUpdatedSeqLen = kCurrentSeqLen + kNewSeqLen;

// EXACT copy of the deployed kernel from
// baselines/L1/problem_071_kv_cache_update_with_rope/cuda_cpp/kernel.cu.
__global__ void problem_071_kv_cache_update_with_rope_kernel(
    const __nv_bfloat16* __restrict__ key_states,
    const __nv_bfloat16* __restrict__ value_states,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    __nv_bfloat16* __restrict__ updated_key_cache,
    __nv_bfloat16* __restrict__ updated_value_cache,
    int B, int H, int current_seq_len, int new_seq_len, int D, int half_D) {

    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= D) return;
    if (b >= B || h >= H) return;

    const long updated_seq_len = current_seq_len + new_seq_len;
    if (s >= updated_seq_len) return;

    const long dst_idx =
        ((static_cast<long>(b) * H + h) * updated_seq_len + s) * D + tid;

    if (s < current_seq_len) {
        const long src_idx =
            ((static_cast<long>(b) * H + h) * current_seq_len + s) * D + tid;
        updated_key_cache[dst_idx] = key_cache[src_idx];
        updated_value_cache[dst_idx] = value_cache[src_idx];
        return;
    }

    const int ns = s - current_seq_len;
    const long new_src_idx =
        ((static_cast<long>(b) * H + h) * new_seq_len + ns) * D + tid;
    const long cos_sin_idx =
        (static_cast<long>(b) * new_seq_len + ns) * D + tid;

    updated_value_cache[dst_idx] = value_states[new_src_idx];

    long rotate_pair_idx;
    float rotate_sign;
    if (tid < half_D) {
        rotate_pair_idx =
            ((static_cast<long>(b) * H + h) * new_seq_len + ns) * D
            + (tid + half_D);
        rotate_sign = -1.0f;
    } else {
        rotate_pair_idx =
            ((static_cast<long>(b) * H + h) * new_seq_len + ns) * D
            + (tid - half_D);
        rotate_sign = 1.0f;
    }

    const float k = __bfloat162float(key_states[new_src_idx]);
    const float c = __bfloat162float(cos[cos_sin_idx]);
    const float si = __bfloat162float(sin[cos_sin_idx]);
    const float rotate_val =
        rotate_sign * __bfloat162float(key_states[rotate_pair_idx]);

    const __nv_bfloat16 term1_bf = __float2bfloat16(k * c);
    const __nv_bfloat16 term2_bf = __float2bfloat16(rotate_val * si);
    const float result =
        __bfloat162float(term1_bf) + __bfloat162float(term2_bf);
    updated_key_cache[dst_idx] = __float2bfloat16(result);
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

    const long key_states_numel =
        static_cast<long>(kBatch) * kNumKvHeads * kNewSeqLen * kHeadDim;
    const long value_states_numel = key_states_numel;
    const long cos_sin_numel =
        static_cast<long>(kBatch) * 1 * kNewSeqLen * kHeadDim;
    const long key_cache_numel =
        static_cast<long>(kBatch) * kNumKvHeads * kCurrentSeqLen * kHeadDim;
    const long value_cache_numel = key_cache_numel;
    const long out_numel =
        static_cast<long>(kBatch) * kNumKvHeads * kUpdatedSeqLen * kHeadDim;

    __nv_bfloat16* d_key_states =
        static_cast<__nv_bfloat16*>(device_alloc<__nv_bfloat16>(key_states_numel));
    __nv_bfloat16* d_value_states =
        static_cast<__nv_bfloat16*>(device_alloc<__nv_bfloat16>(value_states_numel));
    __nv_bfloat16* d_cos =
        static_cast<__nv_bfloat16*>(device_alloc<__nv_bfloat16>(cos_sin_numel));
    __nv_bfloat16* d_sin =
        static_cast<__nv_bfloat16*>(device_alloc<__nv_bfloat16>(cos_sin_numel));
    // key_cache / value_cache may be zero-sized when current_seq_len=0;
    // cudaMalloc(0) is implementation-defined, so guard with a sentinel.
    __nv_bfloat16* d_key_cache = nullptr;
    __nv_bfloat16* d_value_cache = nullptr;
    if (key_cache_numel > 0) {
        d_key_cache =
            static_cast<__nv_bfloat16*>(device_alloc<__nv_bfloat16>(key_cache_numel));
        d_value_cache =
            static_cast<__nv_bfloat16*>(device_alloc<__nv_bfloat16>(value_cache_numel));
    }
    __nv_bfloat16* d_updated_key_cache =
        static_cast<__nv_bfloat16*>(device_alloc<__nv_bfloat16>(out_numel));
    __nv_bfloat16* d_updated_value_cache =
        static_cast<__nv_bfloat16*>(device_alloc<__nv_bfloat16>(out_numel));

    dim3 grid(kUpdatedSeqLen, kNumKvHeads, kBatch);
    dim3 block(kHeadDim);

    for (int it = 0; it < loop_count; ++it) {
        problem_071_kv_cache_update_with_rope_kernel<<<grid, block>>>(
            d_key_states, d_value_states, d_cos, d_sin,
            d_key_cache, d_value_cache,
            d_updated_key_cache, d_updated_value_cache,
            kBatch, kNumKvHeads, kCurrentSeqLen, kNewSeqLen,
            kHeadDim, kHalfHeadDim);
    }
    CUDA_SAFECALL(cudaDeviceSynchronize());

    CUDA_SAFECALL(cudaFree(d_key_states));
    CUDA_SAFECALL(cudaFree(d_value_states));
    CUDA_SAFECALL(cudaFree(d_cos));
    CUDA_SAFECALL(cudaFree(d_sin));
    if (d_key_cache != nullptr) {
        CUDA_SAFECALL(cudaFree(d_key_cache));
        CUDA_SAFECALL(cudaFree(d_value_cache));
    }
    CUDA_SAFECALL(cudaFree(d_updated_key_cache));
    CUDA_SAFECALL(cudaFree(d_updated_value_cache));
    return 0;
}
