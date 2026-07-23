#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// DeepSeek-V3 latent KV expansion with split.
//
// Reference (problem_064_latent_kv_expansion_with_split):
//     num_heads = 128; qk_nope_head_dim = 128; v_head_dim = 128
//     bsz, seq_len, _ = compressed_kv.shape
//     # Stage 1: RMSNorm (float32 reduction, weight applied in f32, cast back to bf16)
//     hidden_states = compressed_kv.to(f32)
//     variance = hidden_states.pow(2).mean(-1, keepdim=True)
//     hidden_states = hidden_states * rsqrt(variance + eps)
//     normalized_kv = (kv_a_layernorm_weight.f32 * hidden_states).to(bf16)
//     # Stage 2: Linear projection (bf16 matmul; weight transposed)
//     expanded_kv = matmul(normalized_kv, kv_b_proj_weight.t())   # [B, S, 32768] bf16
//     # Stage 3: reshape + transpose
//     kv = expanded_kv.view(bsz, seq_len, 128, 256).transpose(1, 2)  # [B, 128, S, 256]
//     # Stage 4: split along last dim
//     k_nope       = kv[:, :, :, :128].contiguous()                  # [B, 128, S, 128]
//     value_states = kv[:, :, :, 128:].contiguous()                  # [B, 128, S, 128]
//
// Per definition.json (insertion order):
//   compressed_kv          : [batch_size, seq_len, kv_lora_rank=512]              bf16
//   kv_a_layernorm_weight  : [kv_lora_rank=512]                                   bf16
//   kv_b_proj_weight       : [kv_expanded_dim=32768, kv_lora_rank=512]            bf16
//   eps                    : scalar                                              f32
//   k_nope                 : [batch_size, num_heads=128, seq_len, qk_nope_head_dim=128]  bf16 (DPS)
//   value_states           : [batch_size, num_heads=128, seq_len, v_head_dim=128]        bf16 (DPS)
//
// eps is a scalar (shape == null in definition.json) so it binds as a C++
// double, NOT a tensor.
//
// Precision contract (CLAUDE.md Rule 8): RMSNorm runs the variance reduction
// and rsqrt in float32, multiplies by the float32-cast weight, and casts the
// normalized result back to bfloat16 BEFORE the projection matmul -- exactly
// the reference cast points. The projection is therefore a bf16 matmul
// (normalized_kv bf16 @ kv_b_proj_weight.t() bf16), routing through the same
// cuBLAS GEMM (transB = T from the explicit .t()) the reference's torch.matmul
// uses. No precision downgrade.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// reshape/transpose/split chain unwinds the projected layout into the
// per-head K_nope / value tensors; the two .contiguous() copies match the
// reference's explicit .contiguous() calls. Outputs written in place into the
// pre-allocated destinations (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& compressed_kv,
    const torch::Tensor& kv_a_layernorm_weight,
    const torch::Tensor& kv_b_proj_weight,
    double eps,
    torch::Tensor k_nope,
    torch::Tensor value_states
) {
    check_bf16_cuda(compressed_kv, "compressed_kv");
    check_bf16_cuda(kv_a_layernorm_weight, "kv_a_layernorm_weight");
    check_bf16_cuda(kv_b_proj_weight, "kv_b_proj_weight");
    check_bf16_cuda(k_nope, "k_nope");
    check_bf16_cuda(value_states, "value_states");
    TORCH_CHECK(compressed_kv.dim() == 3,
                "compressed_kv must be 3D [B, S, kv_lora_rank], got ",
                compressed_kv.dim(), "D");
    TORCH_CHECK(kv_a_layernorm_weight.dim() == 1,
                "kv_a_layernorm_weight must be 1D [kv_lora_rank]");
    TORCH_CHECK(kv_b_proj_weight.dim() == 2,
                "kv_b_proj_weight must be 2D [kv_expanded_dim, kv_lora_rank]");

    const c10::cuda::CUDAGuard device_guard(compressed_kv.device());

    constexpr int64_t kNumHeads = 128;
    constexpr int64_t kQkNopeHeadDim = 128;
    constexpr int64_t kVHeadDim = 128;
    constexpr int64_t kPerHead = kQkNopeHeadDim + kVHeadDim;  // 256

    const int64_t bsz = compressed_kv.size(0);
    const int64_t seq_len = compressed_kv.size(1);

    const auto f32 = torch::kFloat32;

    // Stage 1: RMSNorm in float32; weight applied in float32; cast back to bf16.
    auto hidden_states = compressed_kv.to(f32);                    // [B, S, 512] f32
    auto variance = hidden_states.pow(2).mean(-1, /*keepdim=*/true);
    hidden_states = hidden_states * at::rsqrt(variance + eps);     // eps -> double, broadcasts
    auto normalized_kv =
        (kv_a_layernorm_weight.to(f32) * hidden_states).to(torch::kBFloat16);  // [B, S, 512] bf16

    // Stage 2: bf16 projection. Explicit .t() -> cuBLAS transB = T.
    auto expanded_kv = at::matmul(normalized_kv, kv_b_proj_weight.t());  // [B, S, 32768] bf16

    // Stage 3: reshape to [B, S, 128, 256] then transpose heads/seq -> [B, 128, S, 256].
    auto kv = expanded_kv.view({bsz, seq_len, kNumHeads, kPerHead})
                  .transpose(1, 2);                               // non-contiguous view

    // Stage 4: split along the last dim; each branch gets its own contiguous copy.
    auto k_nope_result =
        kv.narrow(/*dim=*/3, /*start=*/0, /*length=*/kQkNopeHeadDim).contiguous();
    auto value_result =
        kv.narrow(/*dim=*/3, /*start=*/kQkNopeHeadDim, /*length=*/kVHeadDim).contiguous();

    TORCH_CHECK(k_nope_result.sizes() == k_nope.sizes(),
                "computed k_nope shape ", k_nope_result.sizes(),
                " does not match pre-allocated k_nope ", k_nope.sizes());
    TORCH_CHECK(value_result.sizes() == value_states.sizes(),
                "computed value_states shape ", value_result.sizes(),
                " does not match pre-allocated value_states ", value_states.sizes());

    k_nope.copy_(k_nope_result);
    value_states.copy_(value_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "DeepSeek-V3 latent KV expansion with split. Pure ATen/cuBLAS "
          "dispatch: float32 RMSNorm cast back to bf16, one bf16 projection "
          "matmul (normalized_kv @ kv_b_proj_weight.t()), view+transpose to "
          "[B, num_heads=128, S, 256], then split into k_nope / value_states "
          "via narrow+contiguous. Destination-passing style.");
}
