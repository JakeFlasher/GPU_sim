#include "kernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// Strategy (a): two direct at::matmul_out calls, one per output
// stream. Each call goes through PyTorch's cuBLAS dispatch on the
// current CUDA stream, so cuBLAS handles are managed by PyTorch
// (CLAUDE.md Rule 5: cuBLAS is a host API; handles cached by torch,
// stream is current at::cuda::getCurrentCUDAStream()).
//
// Why no custom __global__ kernel: the reference is a 3072x3072
// linear projection with float32 IEEE compute; on sm_120 this is the
// FFMA SM pipe (not tensor cores under FP32 IEEE), and cuBLAS already
// emits the tuned heuristic. A hand-written GEMM matching cuBLAS's
// heuristics is out of scope for a template solution. Per-stream
// matmul saves the reference's concat + slice memcpy traffic.
//
// Note: the (void) stream cast silences -Wunused-parameter; PyTorch
// dispatch picks up the current CUDA stream internally via
// at::cuda::getCurrentCUDAStream() at the call site of at::matmul_out.
void problem_030_flux_concat_split_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& encoder_hidden_states,
    const torch::Tensor& process_weight,
    torch::Tensor& processed_encoder,
    torch::Tensor& processed_hidden,
    cudaStream_t stream) {
    (void)stream;
    auto W_t = process_weight.t();
    at::matmul_out(processed_encoder, encoder_hidden_states, W_t);
    at::matmul_out(processed_hidden, hidden_states, W_t);
}
