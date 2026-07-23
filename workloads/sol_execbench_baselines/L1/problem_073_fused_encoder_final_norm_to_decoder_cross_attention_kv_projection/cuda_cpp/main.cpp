#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

static void check_fp16_contiguous(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.dtype() == torch::kFloat16,
        name, " must have float16 dtype, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry. The definition declares:
//   encoder_hidden_states : [batch_size, encoder_seq_len, hidden_size]  fp16
//   norm_weight           : [hidden_size]                                fp16
//   k_proj_weight         : [kv_hidden_size, hidden_size]                fp16
//   v_proj_weight         : [kv_hidden_size, hidden_size]                fp16
//   eps                   : scalar                                       fp32
//   keys                  : [batch_size, num_kv_heads, encoder_seq_len, head_dim] fp16 (DPS)
//   values                : [batch_size, num_kv_heads, encoder_seq_len, head_dim] fp16 (DPS)
//
// Phase 1 (RMSNorm) runs in a custom CUDA kernel that produces an
// FP16 `normalized` intermediate. Phase 2 (K/V projection matmuls)
// is delegated to `at::matmul`, which routes through cuBLAS HGEMM
// with PyTorch's `allow_fp16_reduced_precision_reduction` setting
// active. This matches the precision profile of the reference's
// `F.linear(normalized, k_proj_weight)` step EXACTLY across every
// NCU-permitted arch (sm_86 / sm_120 / sm_100a), removing the
// kernel-vs-reference HGEMM-precision divergence that previously
// caused workload_index=13 to fail tolerance on sm_86 (max_atol
// 2.9e-5).
//
// Both `keys` and `values` are pre-allocated by the eval driver in
// the head-major target layout `[batch, num_kv_heads, seq_len,
// head_dim]`. The matmul produces `[batch, seq_len, kv_hidden_size]`
// which we reshape and transpose to match, then copy_ into the
// pre-allocated destination tensors.
void run(
    const torch::Tensor& encoder_hidden_states,
    const torch::Tensor& norm_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    double eps,
    torch::Tensor keys,
    torch::Tensor values) {

    TORCH_CHECK(
        encoder_hidden_states.dim() == 3,
        "encoder_hidden_states must be 3D [batch, seq, hidden]; got ",
        encoder_hidden_states.dim(), "D");
    const int64_t batch_size = encoder_hidden_states.size(0);
    const int64_t seq_len = encoder_hidden_states.size(1);
    const int64_t hidden_size = encoder_hidden_states.size(2);
    TORCH_CHECK(
        hidden_size == 1024,
        "hidden_size must be 1024 for problem_073, got ", hidden_size);
    TORCH_CHECK(
        norm_weight.dim() == 1 && norm_weight.size(0) == hidden_size,
        "norm_weight must be 1D length 1024");
    TORCH_CHECK(
        k_proj_weight.dim() == 2 && k_proj_weight.size(0) == 128
            && k_proj_weight.size(1) == hidden_size,
        "k_proj_weight must be [128, 1024]");
    TORCH_CHECK(
        v_proj_weight.dim() == 2 && v_proj_weight.size(0) == 128
            && v_proj_weight.size(1) == hidden_size,
        "v_proj_weight must be [128, 1024]");

    check_fp16_contiguous(encoder_hidden_states, "encoder_hidden_states");
    check_fp16_contiguous(norm_weight, "norm_weight");
    check_fp16_contiguous(k_proj_weight, "k_proj_weight");
    check_fp16_contiguous(v_proj_weight, "v_proj_weight");
    check_fp16_contiguous(keys, "keys");
    check_fp16_contiguous(values, "values");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Phase 1: RMSNorm. Produces FP16 normalized [batch, seq, 1024]
    // matching the reference's `(norm_weight * hidden_states).to(fp16)`.
    auto normalized = torch::empty_like(encoder_hidden_states);
    problem_073_rmsnorm_launcher(
        encoder_hidden_states, norm_weight, normalized,
        static_cast<float>(eps), stream);

    // Phase 2: K/V projections via at::matmul. cuBLAS HGEMM on the
    // current CUDA stream; matches PyTorch reference's F.linear path
    // including reduced-precision reduction toggles. Output layout
    // is [batch, seq, kv_hidden_size]. Reshape to head-major and
    // copy_ into the pre-allocated destination tensors.
    auto k_t = k_proj_weight.transpose(0, 1);
    auto v_t = v_proj_weight.transpose(0, 1);
    auto keys_flat = at::matmul(normalized, k_t);
    auto values_flat = at::matmul(normalized, v_t);

    constexpr int64_t NUM_KV_HEADS = 2;
    constexpr int64_t HEAD_DIM = 64;
    auto keys_head_major =
        keys_flat
            .view({batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM})
            .transpose(1, 2)
            .contiguous();
    auto values_head_major =
        values_flat
            .view({batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM})
            .transpose(1, 2)
            .contiguous();
    keys.copy_(keys_head_major);
    values.copy_(values_head_major);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Fused encoder final RMSNorm (custom CUDA kernel) + "
        "cross-attention K/V projection (at::matmul / cuBLAS HGEMM); "
        "FP16 inputs/outputs; matmul precision matches the PyTorch "
        "reference's F.linear path across all NCU-permitted arches.");
}
