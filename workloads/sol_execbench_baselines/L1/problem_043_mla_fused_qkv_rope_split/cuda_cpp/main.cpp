#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Fused Multi-head Latent Attention (MLA) QKV projection with RoPE split
// (DeepSeek-V3).
//
// Reference (problem_043_mla_fused_qkv_rope_split):
//     num_heads = 128; qk_nope_head_dim = 128; qk_rope_head_dim = 64
//     q_head_dim = 192 (= qk_nope_head_dim + qk_rope_head_dim); kv_lora_rank = 512
//     bsz, seq_len, _ = hidden_states.shape
//
//     # Query pathway
//     q_latent = F.linear(hidden_states, q_a_proj_weight)         # [B, S, 1536] bf16
//     # RMSNorm: float32 variance + rsqrt, normed cast BACK to bf16, then bf16 weight mul
//     input_dtype = q_latent.dtype                                # bf16
//     q_latent_fp32 = q_latent.to(float32)
//     variance = q_latent_fp32.pow(2).mean(-1, keepdim=True)
//     q_latent_normed = q_latent_fp32 * rsqrt(variance + eps)     # f32
//     q_latent = q_a_layernorm_weight * q_latent_normed.to(input_dtype)  # bf16 weight * bf16
//     q = F.linear(q_latent, q_b_proj_weight)                     # [B, S, 24576] bf16
//     q = q.view(bsz, seq_len, num_heads, q_head_dim)             # [B, S, 128, 192]
//     q_nope = q[..., :128].contiguous()                          # [B, S, 128, 128]
//     q_pe   = q[..., 128:].contiguous()                          # [B, S, 128, 64]
//
//     # KV pathway
//     kv_combined = F.linear(hidden_states, kv_a_proj_weight)     # [B, S, 576] bf16
//     compressed_kv = kv_combined[..., :512].contiguous()         # [B, S, 512]
//     k_pe          = kv_combined[..., 512:].contiguous()         # [B, S, 64]
//     k_pe = k_pe.view(bsz, seq_len, 1, 64)                       # [B, S, 1, 64]
//
//     return q_nope, q_pe, compressed_kv, k_pe                    # NO attention
//
// Per definition.json (insertion order):
//   hidden_states         : [batch_size, seq_len, hidden_size=7168]        bf16
//   q_a_proj_weight       : [q_lora_rank=1536, hidden_size=7168]           bf16
//   q_a_layernorm_weight  : [q_lora_rank=1536]                             bf16
//   q_b_proj_weight       : [q_proj_out_dim=24576, q_lora_rank=1536]       bf16
//   kv_a_proj_weight      : [kv_proj_out_dim=576, hidden_size=7168]        bf16
//   rms_norm_eps          : scalar                                         f32
//   q_nope                : [batch_size, seq_len, num_heads=128, qk_nope_head_dim=128]  bf16 (DPS)
//   q_pe                  : [batch_size, seq_len, num_heads=128, qk_rope_head_dim=64]   bf16 (DPS)
//   compressed_kv         : [batch_size, seq_len, kv_lora_rank=512]                     bf16 (DPS)
//   k_pe                  : [batch_size, seq_len, 1, qk_rope_head_dim=64]               bf16 (DPS)
//
// rms_norm_eps is a scalar (shape == null in definition.json) so it binds as a
// C++ double, NOT a tensor.
//
// Precision contract (CLAUDE.md Rule 8): RMSNorm runs the variance reduction and
// rsqrt in float32, casts the normalized result BACK to bfloat16 FIRST, and THEN
// multiplies by the bfloat16 layernorm weight -- exactly the reference cast
// points (`q_a_layernorm_weight * q_latent_normed.to(input_dtype)`; note the
// weight is NOT upcast to f32 here, unlike some sibling RMSNorm ports). Both
// linears are bf16 matmuls (at::linear, no bias) routing through the same cuBLAS
// GEMM (transB = T from F.linear's implicit weight transpose) the reference uses.
// No precision downgrade.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// view/narrow/contiguous chain unwinds the projected layouts into the split
// outputs; the .contiguous() copies match the reference's explicit .contiguous()
// calls. Outputs written in place into the pre-allocated destinations
// (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& q_a_proj_weight,
    const torch::Tensor& q_a_layernorm_weight,
    const torch::Tensor& q_b_proj_weight,
    const torch::Tensor& kv_a_proj_weight,
    double rms_norm_eps,
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor compressed_kv,
    torch::Tensor k_pe
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(q_a_proj_weight, "q_a_proj_weight");
    check_bf16_cuda(q_a_layernorm_weight, "q_a_layernorm_weight");
    check_bf16_cuda(q_b_proj_weight, "q_b_proj_weight");
    check_bf16_cuda(kv_a_proj_weight, "kv_a_proj_weight");
    check_bf16_cuda(q_nope, "q_nope");
    check_bf16_cuda(q_pe, "q_pe");
    check_bf16_cuda(compressed_kv, "compressed_kv");
    check_bf16_cuda(k_pe, "k_pe");
    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden_size], got ",
                hidden_states.dim(), "D");
    TORCH_CHECK(q_a_proj_weight.dim() == 2,
                "q_a_proj_weight must be 2D [q_lora_rank, hidden_size]");
    TORCH_CHECK(q_a_layernorm_weight.dim() == 1,
                "q_a_layernorm_weight must be 1D [q_lora_rank]");
    TORCH_CHECK(q_b_proj_weight.dim() == 2,
                "q_b_proj_weight must be 2D [q_proj_out_dim, q_lora_rank]");
    TORCH_CHECK(kv_a_proj_weight.dim() == 2,
                "kv_a_proj_weight must be 2D [kv_proj_out_dim, hidden_size]");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 128;
    constexpr int64_t kQkNopeHeadDim = 128;
    constexpr int64_t kQkRopeHeadDim = 64;
    constexpr int64_t kQHeadDim = kQkNopeHeadDim + kQkRopeHeadDim;  // 192
    constexpr int64_t kKvLoraRank = 512;

    const int64_t bsz = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);

    const auto f32 = torch::kFloat32;

    // --- Query pathway -----------------------------------------------------
    // q_a_proj: bf16 linear (no bias). at::linear(x, W) == x @ W.t().
    auto q_latent = at::linear(hidden_states, q_a_proj_weight);  // [B, S, 1536] bf16

    // RMSNorm: f32 variance + rsqrt; normed cast BACK to bf16 FIRST; then the
    // bf16 layernorm weight multiply (weight is NOT upcast to f32 -- matches
    // `q_a_layernorm_weight * q_latent_normed.to(input_dtype)`).
    auto q_latent_fp32 = q_latent.to(f32);                       // [B, S, 1536] f32
    auto variance = q_latent_fp32.pow(2).mean(-1, /*keepdim=*/true);
    auto q_latent_normed = q_latent_fp32 * at::rsqrt(variance + rms_norm_eps);  // f32
    q_latent = q_a_layernorm_weight * q_latent_normed.to(torch::kBFloat16);     // bf16

    // q_b_proj: bf16 linear (no bias) -> [B, S, 24576], then view to heads.
    auto q = at::linear(q_latent, q_b_proj_weight);              // [B, S, 24576] bf16
    q = q.view({bsz, seq_len, kNumHeads, kQHeadDim});            // [B, S, 128, 192]

    // Split along last dim into nope / pe; each branch its own contiguous copy.
    auto q_nope_result =
        q.narrow(/*dim=*/3, /*start=*/0, /*length=*/kQkNopeHeadDim).contiguous();
    auto q_pe_result =
        q.narrow(/*dim=*/3, /*start=*/kQkNopeHeadDim, /*length=*/kQkRopeHeadDim).contiguous();

    // --- KV pathway --------------------------------------------------------
    // kv_a_proj: bf16 linear (no bias) -> [B, S, 576].
    auto kv_combined = at::linear(hidden_states, kv_a_proj_weight);  // [B, S, 576] bf16

    auto compressed_kv_result =
        kv_combined.narrow(/*dim=*/2, /*start=*/0, /*length=*/kKvLoraRank).contiguous();  // [B, S, 512]
    auto k_pe_result =
        kv_combined.narrow(/*dim=*/2, /*start=*/kKvLoraRank, /*length=*/kQkRopeHeadDim).contiguous();  // [B, S, 64]
    k_pe_result = k_pe_result.view({bsz, seq_len, 1, kQkRopeHeadDim});  // [B, S, 1, 64]

    TORCH_CHECK(q_nope_result.sizes() == q_nope.sizes(),
                "computed q_nope shape ", q_nope_result.sizes(),
                " does not match pre-allocated q_nope ", q_nope.sizes());
    TORCH_CHECK(q_pe_result.sizes() == q_pe.sizes(),
                "computed q_pe shape ", q_pe_result.sizes(),
                " does not match pre-allocated q_pe ", q_pe.sizes());
    TORCH_CHECK(compressed_kv_result.sizes() == compressed_kv.sizes(),
                "computed compressed_kv shape ", compressed_kv_result.sizes(),
                " does not match pre-allocated compressed_kv ", compressed_kv.sizes());
    TORCH_CHECK(k_pe_result.sizes() == k_pe.sizes(),
                "computed k_pe shape ", k_pe_result.sizes(),
                " does not match pre-allocated k_pe ", k_pe.sizes());

    q_nope.copy_(q_nope_result);
    q_pe.copy_(q_pe_result);
    compressed_kv.copy_(compressed_kv_result);
    k_pe.copy_(k_pe_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Fused MLA QKV projection with RoPE split (DeepSeek-V3). Pure "
          "ATen/cuBLAS dispatch: q_a_proj linear -> float32 RMSNorm cast back "
          "to bf16 then bf16 weight mul -> q_b_proj linear -> view "
          "[B, S, 128, 192] and split into q_nope / q_pe; kv_a_proj linear -> "
          "split into compressed_kv / k_pe (k_pe viewed to [B, S, 1, 64]). "
          "Splits via narrow+contiguous. Destination-passing style.");
}
