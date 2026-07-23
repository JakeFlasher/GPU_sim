#include "kernel.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// One pure-boolean CUDA kernel that writes both the full causal
// attention mask and the sliding-window attention mask in a single
// pass over (b, h, t, j).
//
// Per-element math (matching the reference's two `masked_fill`
// operations exactly):
//
//   Full mask:
//     target_idx  = t
//     source_idx  = j - past_kv_length
//     causal_cond = (target_idx >= source_idx)
//     full_mask = torch.ones(...).masked_fill(causal_cond, False)
//        => full_attention_mask[b, h, t, j] = !causal_cond
//
//   Sliding-window mask:
//     swa_mask = torch.zeros(...).masked_fill(valid_positions, False)
//     This is a NO-OP: a buffer of False values filled with False at
//     the True positions yields a buffer of False values. The
//     reference's `valid_positions` computation has no observable
//     effect on the returned tensor. To match the canonical
//     reference bit-for-bit, the kernel writes `False` for every
//     (b, h, t, j) in sliding_window_attention_mask. The
//     reference's combine-with-window branch is therefore dead
//     code under the canonical workload contract.
//
// Kernel topology:
//   * Grid (B, num_heads, T). One CTA per (b, h, t) row.
//   * BLOCK_THREADS = up to 256; stride-loops over `total_length`.
//   * Each thread computes both bool outputs at (b, h, t, j) and
//     writes them via the contiguous row offsets.

namespace {

constexpr int kBlockThreads = 256;

__global__ void __launch_bounds__(kBlockThreads, 4)
    k_hybrid_attention_mask_preparation(
        bool* __restrict__ full_attention_mask,
        bool* __restrict__ sliding_window_attention_mask,
        int H,
        int T,
        int total_length,
        int past_kv_length
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    const int tid = threadIdx.x;

    const size_t row_offset = (((size_t)b * H + h) * T
                               + (size_t)t) * total_length;
    bool* __restrict__ full_row = full_attention_mask + row_offset;
    bool* __restrict__ swa_row =
        sliding_window_attention_mask + row_offset;

    const int target_idx = t;

    for (int j = tid; j < total_length; j += kBlockThreads) {
        const int source_idx = j - past_kv_length;
        const bool causal_cond = (target_idx >= source_idx);
        full_row[j] = !causal_cond;
        swa_row[j] = false;  // see header comment: reference's
                             // masked_fill on a zeros buffer with False
                             // is a no-op; the kernel emits the same.
    }
}

}  // namespace

void problem_028_hybrid_attention_mask_preparation_launcher(
    int64_t batch_size,
    int64_t seq_length,
    int64_t past_kv_length,
    int64_t sliding_window,
    torch::Tensor& full_attention_mask,
    torch::Tensor& sliding_window_attention_mask,
    cudaStream_t stream
) {
    const int B = static_cast<int>(batch_size);
    const int H = static_cast<int>(full_attention_mask.size(1));
    const int T = static_cast<int>(seq_length);
    const int total_length = static_cast<int>(seq_length
                                              + past_kv_length);

    dim3 grid(B, H, T);
    dim3 block(kBlockThreads);

    (void)sliding_window;  // see kernel comment: sliding-window mask is
                           // all-False under the reference's no-op
                           // masked_fill, so the kernel doesn't read
                           // sliding_window. The host launcher accepts
                           // it for runtime-check / API completeness.

    k_hybrid_attention_mask_preparation<<<grid, block, 0, stream>>>(
        full_attention_mask.data_ptr<bool>(),
        sliding_window_attention_mask.data_ptr<bool>(),
        H,
        T,
        total_length,
        static_cast<int>(past_kv_length)
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "problem_028 hybrid attention mask preparation kernel "
                "launch failed");
}
