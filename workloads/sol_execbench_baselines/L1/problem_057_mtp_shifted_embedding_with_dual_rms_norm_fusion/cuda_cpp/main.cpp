#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Multi-Token Prediction (MTP) shifted embedding with dual RMSNorm fusion
// (inclusionAI/Ling-flash-2.0).
//
// Reference (problem_057_mtp_shifted_embedding_with_dual_rms_norm_fusion):
//     # Stage 1: Roll input_ids by -1 along seq; last position -> 0.
//     shifted_input_ids = torch.roll(input_ids, shifts=-1, dims=-1)
//     shifted_input_ids[:, -1] = 0
//     # Stage 2: Embed shifted ids (gather rows of word_embeddings).
//     input_embeds = word_embeddings[shifted_input_ids]          # [B, S, H] bf16
//     # Stage 3: RMSNorm the embeddings (enorm). f32 variance + rsqrt, normed cast
//     #         BACK to bf16, then bf16 enorm_weight mul.
//     input_embeds_fp32 = input_embeds.to(float32)
//     embed_variance = input_embeds_fp32.pow(2).mean(-1, keepdim=True)
//     input_embeds_normed = input_embeds_fp32 * rsqrt(embed_variance + eps)   # f32
//     input_embeds_normed = enorm_weight * input_embeds_normed.to(enorm_weight.dtype)  # bf16
//     # Stage 4: RMSNorm the hidden states (hnorm), same cast points, bf16 hnorm_weight.
//     hidden_states_fp32 = hidden_states.to(float32)
//     hidden_variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
//     hidden_states_normed = hidden_states_fp32 * rsqrt(hidden_variance + eps) # f32
//     hidden_states_normed = hnorm_weight * hidden_states_normed.to(hnorm_weight.dtype)  # bf16
//     # Stage 5: Concatenate along last dim -- enorm (embeds) FIRST, hnorm (hidden) SECOND.
//     concatenated = torch.cat([input_embeds_normed, hidden_states_normed], dim=-1)  # [B, S, 2H]
//     # Stage 6: Project back to hidden_size.
//     fused_hidden_states = torch.matmul(concatenated, eh_proj_weight.t())    # [B, S, H]
//     return fused_hidden_states
//
// Per definition.json (insertion order):
//   input_ids        : [batch_size, seq_len]                          int64
//   hidden_states    : [batch_size, seq_len, hidden_size=4096]        bf16
//   word_embeddings  : [vocab_size=157184, hidden_size=4096]          bf16
//   enorm_weight     : [hidden_size=4096]                             bf16
//   hnorm_weight     : [hidden_size=4096]                             bf16
//   eh_proj_weight   : [hidden_size=4096, hidden_size_x2=8192]        bf16
//   rms_norm_eps     : scalar                                         f32
//   fused_hidden_states : [batch_size, seq_len, hidden_size=4096]     bf16 (DPS)
//
// rms_norm_eps is a scalar (shape == null in definition.json) so it binds as a
// C++ double, NOT a tensor.
//
// Precision contract (CLAUDE.md Rule 8): both RMSNorms run the variance reduction
// and rsqrt in float32, cast the normalized result BACK to bfloat16 FIRST, and
// THEN multiply by the bf16 norm weight -- exactly the reference cast points
// (`weight * normed.to(weight.dtype)`; the weight is bf16, not upcast to f32).
// The concatenation is therefore bf16, and the projection is a bf16 matmul
// (concat bf16 @ eh_proj_weight.t() bf16) routing through the same cuBLAS GEMM
// (transB = T from the explicit .t()) the reference's torch.matmul uses. No
// precision downgrade.
//
// Roll semantics: torch.roll returns a NEW tensor (input_ids is not mutated);
// `shifted_input_ids[:, -1] = 0` then zeroes the last seq position of that copy.
// at::roll likewise returns a fresh tensor, so zeroing its last slice in place is
// faithful and does not touch the caller's input_ids. Embedding gather
// `word_embeddings[ids]` == at::embedding(word_embeddings, ids).
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. Outputs
// written in place into the pre-allocated destination (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& input_ids,
    const torch::Tensor& hidden_states,
    const torch::Tensor& word_embeddings,
    const torch::Tensor& enorm_weight,
    const torch::Tensor& hnorm_weight,
    const torch::Tensor& eh_proj_weight,
    double rms_norm_eps,
    torch::Tensor fused_hidden_states
) {
    TORCH_CHECK(input_ids.is_cuda(), "input_ids must be a CUDA tensor");
    TORCH_CHECK(input_ids.dtype() == torch::kInt64,
                "input_ids must be int64, got ", input_ids.dtype());
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(word_embeddings, "word_embeddings");
    check_bf16_cuda(enorm_weight, "enorm_weight");
    check_bf16_cuda(hnorm_weight, "hnorm_weight");
    check_bf16_cuda(eh_proj_weight, "eh_proj_weight");
    check_bf16_cuda(fused_hidden_states, "fused_hidden_states");
    TORCH_CHECK(input_ids.dim() == 2,
                "input_ids must be 2D [B, S], got ", input_ids.dim(), "D");
    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden_size], got ",
                hidden_states.dim(), "D");
    TORCH_CHECK(word_embeddings.dim() == 2,
                "word_embeddings must be 2D [vocab_size, hidden_size]");
    TORCH_CHECK(enorm_weight.dim() == 1, "enorm_weight must be 1D [hidden_size]");
    TORCH_CHECK(hnorm_weight.dim() == 1, "hnorm_weight must be 1D [hidden_size]");
    TORCH_CHECK(eh_proj_weight.dim() == 2,
                "eh_proj_weight must be 2D [hidden_size, hidden_size_x2]");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t seq_len = input_ids.size(1);

    const auto f32 = torch::kFloat32;

    // Stage 1: roll input_ids by -1 along the seq dim, then zero the last position.
    // at::roll returns a fresh tensor; zeroing its last seq slice in place does
    // not mutate the caller's input_ids.
    auto shifted_input_ids = at::roll(input_ids, /*shifts=*/{-1}, /*dims=*/{-1});
    shifted_input_ids.narrow(/*dim=*/1, /*start=*/seq_len - 1, /*length=*/1).zero_();

    // Stage 2: embed shifted ids (gather rows of word_embeddings).
    auto input_embeds = at::embedding(word_embeddings, shifted_input_ids);  // [B, S, H] bf16

    // Stage 3: RMSNorm the embeddings (enorm). f32 variance + rsqrt; cast normed
    // back to bf16 FIRST; then bf16 enorm_weight multiply.
    auto input_embeds_fp32 = input_embeds.to(f32);
    auto embed_variance = input_embeds_fp32.pow(2).mean(-1, /*keepdim=*/true);
    auto input_embeds_normed = input_embeds_fp32 * at::rsqrt(embed_variance + rms_norm_eps);  // f32
    auto enorm_out = enorm_weight * input_embeds_normed.to(torch::kBFloat16);                 // bf16

    // Stage 4: RMSNorm the hidden states (hnorm). Same cast points, bf16 hnorm_weight.
    auto hidden_states_fp32 = hidden_states.to(f32);
    auto hidden_variance = hidden_states_fp32.pow(2).mean(-1, /*keepdim=*/true);
    auto hidden_states_normed = hidden_states_fp32 * at::rsqrt(hidden_variance + rms_norm_eps);  // f32
    auto hnorm_out = hnorm_weight * hidden_states_normed.to(torch::kBFloat16);                   // bf16

    // Stage 5: concatenate along last dim -- enorm (embeds) FIRST, hnorm SECOND.
    auto concatenated = at::cat({enorm_out, hnorm_out}, /*dim=*/-1);  // [B, S, 2H] bf16

    // Stage 6: bf16 projection. Explicit .t() -> cuBLAS transB = T.
    auto fused_result = at::matmul(concatenated, eh_proj_weight.t());  // [B, S, H] bf16

    TORCH_CHECK(fused_result.sizes() == fused_hidden_states.sizes(),
                "computed fused_hidden_states shape ", fused_result.sizes(),
                " does not match pre-allocated fused_hidden_states ",
                fused_hidden_states.sizes());

    fused_hidden_states.copy_(fused_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "MTP shifted embedding with dual RMSNorm fusion (Ling-flash-2.0). "
          "Pure ATen/cuBLAS dispatch: roll input_ids by -1 along seq (last pos "
          "-> 0), at::embedding gather, float32 dual RMSNorm (enorm on embeds, "
          "hnorm on hidden_states) cast back to bf16 then bf16 weight mul, "
          "cat([enorm_out, hnorm_out], dim=-1), then bf16 projection "
          "matmul(concat, eh_proj_weight.t()). Destination-passing style.");
}
