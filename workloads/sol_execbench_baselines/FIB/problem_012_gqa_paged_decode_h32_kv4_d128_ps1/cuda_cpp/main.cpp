#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "kernel.h"

// Destination-passing-style entry point matching the
// reference shape contract at
// references/NVIDIA_Manuals/solbench_problems/problem_012_gqa_paged_decode_h32_kv4_d128_ps1.md.
//
//   q          [B, H_q=32, D=128]  bf16
//   k_cache    [num_pages, page_size=1, H_kv=4, D=128]  bf16
//   v_cache    same shape as k_cache
//   kv_indptr  [B+1]  int32
//   kv_indices [N]    int32
//   sm_scale   scalar float32
//   output     [B, H_q, D]  bf16   (in-place)
//   lse        [B, H_q]     fp32   (in-place; base-2 log-sum-exp)
void run(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    double sm_scale,
    torch::Tensor output,
    torch::Tensor lse) {

    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k_cache.is_cuda(), "k_cache must be CUDA");
    TORCH_CHECK(v_cache.is_cuda(), "v_cache must be CUDA");
    TORCH_CHECK(kv_indptr.is_cuda(), "kv_indptr must be CUDA");
    TORCH_CHECK(kv_indices.is_cuda(), "kv_indices must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(lse.is_cuda(), "lse must be CUDA");

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k_cache.dtype() == torch::kBFloat16, "k_cache must be bf16");
    TORCH_CHECK(v_cache.dtype() == torch::kBFloat16, "v_cache must be bf16");
    TORCH_CHECK(kv_indptr.dtype() == torch::kInt32, "kv_indptr must be int32");
    TORCH_CHECK(kv_indices.dtype() == torch::kInt32, "kv_indices must be int32");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bf16");
    TORCH_CHECK(lse.dtype() == torch::kFloat32, "lse must be fp32");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
    TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
    TORCH_CHECK(kv_indptr.is_contiguous(), "kv_indptr must be contiguous");
    TORCH_CHECK(kv_indices.is_contiguous(), "kv_indices must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");

    TORCH_CHECK(q.dim() == 3, "q must be 3D");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must be 4D");
    TORCH_CHECK(kv_indptr.dim() == 1, "kv_indptr must be 1D");
    TORCH_CHECK(kv_indices.dim() == 1, "kv_indices must be 1D");
    TORCH_CHECK(output.dim() == 3, "output must be 3D");
    TORCH_CHECK(lse.dim() == 2, "lse must be 2D");

    const int64_t B = q.size(0);
    const int64_t H_q = q.size(1);
    const int64_t D = q.size(2);
    TORCH_CHECK(H_q == 32, "num_qo_heads must equal 32");
    TORCH_CHECK(D == 128, "head_dim must equal 128");

    TORCH_CHECK(k_cache.size(1) == 1, "page_size must equal 1");
    const int64_t H_kv = k_cache.size(2);
    TORCH_CHECK(H_kv == 4, "num_kv_heads must equal 4");
    TORCH_CHECK(k_cache.size(3) == D, "k_cache head_dim mismatch");
    TORCH_CHECK(v_cache.size(0) == k_cache.size(0)
                && v_cache.size(1) == 1
                && v_cache.size(2) == H_kv
                && v_cache.size(3) == D,
                "v_cache shape mismatch with k_cache");

    TORCH_CHECK(kv_indptr.size(0) == B + 1,
                "kv_indptr length must equal batch_size + 1");
    TORCH_CHECK(output.size(0) == B && output.size(1) == H_q
                && output.size(2) == D,
                "output shape mismatch");
    TORCH_CHECK(lse.size(0) == B && lse.size(1) == H_q,
                "lse shape mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_012_gqa_paged_decode_launcher(
        q, k_cache, v_cache, kv_indptr, kv_indices,
        static_cast<float>(sm_scale),
        output, lse, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "GQA paged decode (problem_012); destination-passing-style; "
        "online softmax; FP32 accumulators with BF16 inputs/outputs.");
}
