#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

#include <cstdint>
#include <string>

static void check_f32_cuda_contiguous(
    const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// DPS entry point. Reference (sol-execbench L2/030):
//   concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
//   processed    = torch.matmul(concatenated, process_weight.t())
//   processed_encoder = processed[:, :text_seq_len, :]
//   processed_hidden  = processed[:, text_seq_len:, :]
//
// Strategy (a): the linear projection is the same for both streams, so
// we compute each stream's projection directly into its destination
// buffer via at::matmul_out. The reference's concat + slice memcpy
// traffic is eliminated; cuBLAS launches twice instead of once. The
// kernel beats the reference on large workloads (where the saved
// memcpy dominates) and trails on small workloads (where the second
// cuBLAS launch overhead dominates). Speedup on every workload would
// require a hand-written fused TF32 GEMM matching cuBLAS heuristics;
// per the project's correctness-only landing contract (CLAUDE.md
// §"Acceptance contract for *.cu template solutions"), local speedup
// is diagnostic only.
void run(
    torch::Tensor hidden_states,
    torch::Tensor encoder_hidden_states,
    torch::Tensor process_weight,
    torch::Tensor processed_encoder,
    torch::Tensor processed_hidden) {
    check_f32_cuda_contiguous(hidden_states, "hidden_states");
    check_f32_cuda_contiguous(
        encoder_hidden_states, "encoder_hidden_states");
    check_f32_cuda_contiguous(process_weight, "process_weight");
    check_f32_cuda_contiguous(processed_encoder, "processed_encoder");
    check_f32_cuda_contiguous(processed_hidden, "processed_hidden");

    TORCH_CHECK(
        hidden_states.dim() == 3, "hidden_states must be 3D [B, S, D]");
    TORCH_CHECK(
        encoder_hidden_states.dim() == 3,
        "encoder_hidden_states must be 3D [B, S, D]");
    TORCH_CHECK(
        process_weight.dim() == 2, "process_weight must be 2D [D, D]");
    TORCH_CHECK(
        processed_encoder.dim() == 3, "processed_encoder must be 3D");
    TORCH_CHECK(
        processed_hidden.dim() == 3, "processed_hidden must be 3D");

    const int64_t hidden_dim = process_weight.size(0);
    TORCH_CHECK(
        hidden_dim == 3072,
        "hidden_dim must be 3072, got ", hidden_dim);
    TORCH_CHECK(
        process_weight.size(1) == hidden_dim,
        "process_weight must be square [3072, 3072]");
    TORCH_CHECK(
        hidden_states.size(2) == hidden_dim,
        "hidden_states last dim must be 3072, got ", hidden_states.size(2));
    TORCH_CHECK(
        encoder_hidden_states.size(2) == hidden_dim,
        "encoder_hidden_states last dim must be 3072, got ",
        encoder_hidden_states.size(2));

    const int64_t batch = hidden_states.size(0);
    const int64_t img_seq_len = hidden_states.size(1);
    const int64_t text_seq_len = encoder_hidden_states.size(1);
    TORCH_CHECK(
        encoder_hidden_states.size(0) == batch,
        "encoder_hidden_states batch dim must match");
    TORCH_CHECK(
        processed_encoder.size(0) == batch
            && processed_encoder.size(1) == text_seq_len
            && processed_encoder.size(2) == hidden_dim,
        "processed_encoder must be [B, text_seq_len, 3072]");
    TORCH_CHECK(
        processed_hidden.size(0) == batch
            && processed_hidden.size(1) == img_seq_len
            && processed_hidden.size(2) == hidden_dim,
        "processed_hidden must be [B, img_seq_len, 3072]");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_030_flux_concat_split_launcher(
        hidden_states,
        encoder_hidden_states,
        process_weight,
        processed_encoder,
        processed_hidden,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "FLUX concat-then-project-then-split pattern: per-stream "
        "at::matmul_out (cuBLAS dispatch on the current CUDA stream). "
        "float32, destination-passing-style.");
}
