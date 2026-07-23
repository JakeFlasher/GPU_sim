#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Two-phase MoE accumulation:
//   Phase 1: copy final_hidden_states → output (cudaMemcpyAsync on the
//            current stream).
//   Phase 2: for each (selected_token_idx i, hidden lane h):
//              atomicAdd(&output[token_indices[i] * H + h],
//                        expert_outputs[i * H + h])
//
// hidden_size=3072 is pinned by the problem definition. We tile the
// hidden axis across blockDim.x and use a 2D grid (selected_tokens,
// H_tiles). atomicAdd on `__nv_bfloat16` is available on compute
// capability >= 8.0 (sm_80+), which covers both the local 3090 Ti
// (sm_86) and the Phase-2 B200 (sm_100a) target.

namespace {

constexpr int HIDDEN_SIZE = 3072;
constexpr int BLOCK_THREADS = 256;

__global__ void __launch_bounds__(BLOCK_THREADS, 8)
index_add_kernel(
    const __nv_bfloat16* __restrict__ expert_outputs,
    const long* __restrict__ token_indices,
    __nv_bfloat16* __restrict__ output,
    int num_selected) {

    const int sel = blockIdx.x;
    if (sel >= num_selected) return;

    const int h_base = blockIdx.y * BLOCK_THREADS + threadIdx.x;
    if (h_base >= HIDDEN_SIZE) return;

    const long tok = token_indices[sel];
    const long src_off = static_cast<long>(sel) * HIDDEN_SIZE + h_base;
    const long dst_off = tok * HIDDEN_SIZE + h_base;

    const __nv_bfloat16 v = expert_outputs[src_off];
    atomicAdd(output + dst_off, v);
}

}  // namespace

void problem_008_expert_output_weighted_index_add_accumulation_launcher(
    const torch::Tensor& final_hidden_states,
    const torch::Tensor& expert_outputs,
    const torch::Tensor& token_indices,
    torch::Tensor& output,
    cudaStream_t stream) {

    const int batch_seq_len = static_cast<int>(final_hidden_states.size(0));
    const int H = static_cast<int>(final_hidden_states.size(1));
    const int num_selected = static_cast<int>(expert_outputs.size(0));
    (void)H;  // pinned at HIDDEN_SIZE; checked in main.cpp

    // Phase 1: byte-exact copy of the accumulation buffer into output.
    cudaMemcpyAsync(
        output.data_ptr<at::BFloat16>(),
        final_hidden_states.data_ptr<at::BFloat16>(),
        static_cast<size_t>(batch_seq_len) * HIDDEN_SIZE * sizeof(at::BFloat16),
        cudaMemcpyDeviceToDevice,
        stream);

    // Phase 2: atomic scatter-add expert_outputs into output rows
    // indexed by token_indices.
    const int h_tiles = (HIDDEN_SIZE + BLOCK_THREADS - 1) / BLOCK_THREADS;
    dim3 grid(num_selected, h_tiles);
    dim3 block(BLOCK_THREADS);
    index_add_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            expert_outputs.data_ptr<at::BFloat16>()),
        token_indices.data_ptr<long>(),
        reinterpret_cast<__nv_bfloat16*>(
            output.data_ptr<at::BFloat16>()),
        num_selected);
}
