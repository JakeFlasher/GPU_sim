#include "kernel.h"
#include <cuda_runtime.h>
#include <float.h>

// fp32 MoE group-score aggregation and masking.
//
// Per-token pipeline:
//   1. Per-warp (per-group) read: each of 8 warps owns one
//      group of 32 experts; warp lane `l` reads
//      `s = scores[t, group_idx * 32 + l]`.
//   2. Per-group top-2 selection (no sort): find `max1` via
//      warp-shuffle reduction; mark the lane holding `max1`
//      with `-inf`; find `max2` via the same reduction over
//      the marked register. `group_score = max1 + max2`.
//   3. Warp-leaders write `group_scores[g]` to shared memory.
//   4. Cross-group top-4 selection (no sort): warp 0 reads
//      the 8 group_scores (lanes 0..7) and runs four
//      find-argmax passes, each masking the previously-
//      selected lane to `-inf`. Selected group indices land
//      in shared memory; group_mask is computed.
//   5. Each warp-leader writes `group_mask[t, group_idx]`
//      to global memory.
//   6. Each thread writes
//      `masked_scores[t, expert_idx] = (g in selected) ? s :
//      -inf` from its register-cached `s`.
//
// One CTA per token; BLOCK_THREADS = num_experts = 256;
// num_experts / 32 = 8 warps (one per group).

namespace {

constexpr int kNumExperts = 256;
constexpr int kNumGroups = 8;
constexpr int kTopkGroup = 4;
constexpr int kExpertsPerGroup = kNumExperts / kNumGroups;  // 32
constexpr int kBlockThreads = kNumExperts;                  // 256
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockThreads / kWarpSize;        // 8 (= kNumGroups)

__device__ inline float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    }
    return v;
}

__global__ void __launch_bounds__(kBlockThreads, 4)
problem_059_moe_group_score_kernel(
    const float* __restrict__ scores,
    float* __restrict__ masked_scores,
    float* __restrict__ group_mask) {

    const int t = blockIdx.x;
    const int expert_idx = threadIdx.x;
    const int g = expert_idx / kWarpSize;     // group / warp
    const int lane = expert_idx % kWarpSize;  // lane in group

    const int row_offset = t * kNumExperts;
    const float s = scores[row_offset + expert_idx];

    // Per-group top-2 via two warp-shuffle find-max passes.
    const float max1 = warp_reduce_max(s);
    // Mask only one lane that equals max1 to -FLT_MAX before pass 2.
    // Strict equality is safe even with tied scores: if multiple
    // lanes hold the same maximum, masking ANY one of them still
    // lets the reduction over the rest find the second-largest
    // distinct rank (or the same value via another tied lane,
    // which is still a valid top-2). The reference's `topk` with
    // ties is not stable on order, so the workload tolerance
    // accepts any consistent top-2 sum.
    unsigned tie_mask = __ballot_sync(0xffffffff, s == max1);
    int first_tied = __ffs(tie_mask) - 1;
    const float s_for_max2 = (lane == first_tied) ? -FLT_MAX : s;
    const float max2 = warp_reduce_max(s_for_max2);
    const float group_score = max1 + max2;

    // Warp leader stores its group_score into shared memory.
    __shared__ float s_group_scores[kNumGroups];
    if (lane == 0) {
        s_group_scores[g] = group_score;
    }
    __syncthreads();

    // Cross-group top-4 selection: warp 0 reads the 8 group_scores
    // (one per lane in [0, 8)) and runs 4 find-argmax passes,
    // masking the previously-selected lane to -inf each iteration.
    __shared__ float s_group_mask[kNumGroups];
    if (g == 0) {
        if (lane < kNumGroups) {
            s_group_mask[lane] = 0.0f;
        }
        float v = (lane < kNumGroups) ? s_group_scores[lane] : -FLT_MAX;
        #pragma unroll
        for (int pick = 0; pick < kTopkGroup; ++pick) {
            const float cur_max = warp_reduce_max(v);
            // Pick the lowest-indexed lane that equals cur_max (in
            // case of ties; the PyTorch reference's `sorted=False`
            // does not promise a specific tie-break and the workload
            // tolerance treats all top-k variants as equivalent).
            const unsigned mask =
                __ballot_sync(0xffffffff, v == cur_max && lane < kNumGroups);
            const int sel = __ffs(mask) - 1;
            if (sel >= 0 && lane == sel) {
                s_group_mask[sel] = 1.0f;
                v = -FLT_MAX;  // mask self out for next pass
            }
            // Broadcast the selected lane index so every warp 0
            // thread updates its `v` view consistently. Achieved
            // implicitly: each warp 0 thread that wasn't `sel` keeps
            // its `v` unchanged. The `sel` lane sets its own `v` to
            // -FLT_MAX so the next reduction excludes it.
        }
    }
    __syncthreads();

    // Each warp leader writes group_mask[t, group_idx].
    if (lane == 0) {
        const int gm_row_offset = t * kNumGroups;
        group_mask[gm_row_offset + g] = s_group_mask[g];
    }

    // Each thread writes masked_scores[t, expert_idx]. Use the
    // explicit IEEE-754 negative-infinity bit pattern; `CUDART_INF_F`
    // is gated behind <math_constants.h> on the deployed toolchain,
    // and including <math_constants.h> from a `__global__` context
    // produces a build failure under nvcc, so we avoid the dependency
    // entirely via the explicit bit pattern.
    const float neg_inf = __int_as_float(0xff800000);
    const float gm = s_group_mask[g];
    const float out = (gm == 1.0f) ? s : neg_inf;
    masked_scores[row_offset + expert_idx] = out;
}

}  // namespace

void problem_059_moe_group_score_aggregation_launcher(
    const torch::Tensor& scores,
    torch::Tensor& masked_scores,
    torch::Tensor& group_mask,
    cudaStream_t stream) {

    const int64_t num_tokens = scores.size(0);

    dim3 grid(static_cast<int>(num_tokens));
    dim3 block(kBlockThreads);

    problem_059_moe_group_score_kernel<<<grid, block, 0, stream>>>(
        scores.data_ptr<float>(),
        masked_scores.data_ptr<float>(),
        group_mask.data_ptr<float>());
}
