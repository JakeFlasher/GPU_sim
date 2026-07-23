#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <string>

static void check_bf16_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the fp32-inner attention
// Q@K^T computation with GQA-style key view and derived scaling.
// Per definition.json:
//   query        : [B, num_attention_heads=32, S, head_dim=128]   bf16
//   key          : [B, num_key_value_heads=32, S, head_dim=128]   bf16
//   attn_weights : [B, num_attention_heads=32, S, S]              bf16 DPS
//
// Algorithm (mirrors the reference exactly):
//   head_dim = 128
//   num_key_value_groups = 1
//   scaling = head_dim ** -0.5  # derived constant, NOT an input
//   key_expanded = key.unsqueeze(2).expand(B, num_key_value_heads,
//       num_key_value_groups, S, head_dim)
//   key_states = key_expanded.reshape(B, num_attention_heads, S,
//       head_dim)
//     # n_rep = num_key_value_groups = 1: the GQA "repetition" is
//     # a no-op view; the singleton expand axis flattens trivially
//     # through reshape.
//   q_fp32 = query.to(fp32)
//   k_fp32_t = key_states.transpose(2, 3).to(fp32)
//     # logically [B, H, head_dim, S] but the .transpose(2, 3) is
//     # a stride-only operation; .to(fp32) materializes a fp32
//     # tensor preserving the transposed-view stride pattern, so
//     # the physical fp32 storage is laid out as [B, H, S,
//     # head_dim] (row-major contiguous) while the logical shape
//     # reads as [B, H, head_dim, S].
//   attn_weights_fp32 = at::matmul(q_fp32, k_fp32_t)
//     # [B, H, S, S] fp32. PyTorch's 4D fp32 at::matmul forwards
//     # to cublasSgemmStridedBatched (legacy non-Ex fp32 API)
//     # with transA=T (key operand transposed), transB=N (query
//     # operand normal), m=n=S, k=head_dim, lda=ldb=head_dim,
//     # ldc=S, strideA=strideB=S*head_dim, strideC=S*S,
//     # batch_count=B*H.
//   attn_weights_fp32 = attn_weights_fp32 * scaling
//   attn_weights.copy_(attn_weights_fp32.to(bf16))
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   num_attention_heads  = 32
//   num_key_value_heads  = 32
//   head_dim             = 128
//   num_key_value_groups = num_attention_heads / num_key_value_heads = 1
//
// `scaling` is a derived constant computed inside `run()` from the
// const `head_dim` — NOT an input scalar. The binding signature
// takes only (query, key, attn_weights). Differs from problem_049,
// which receives `double scaling` as a Python-float input.
//
// Implementation: pure ATen/cuBLAS dispatch — NO custom CUDA
// kernels. The fp32 batched matmul dispatches
// cublasSgemmStridedBatched (the legacy non-Ex fp32-only API)
// with transA=T, transB=N — the SAME already-verified dispatch
// surface as problem_049_attention_qk_matmul_with_gqa_repeat_
// and_scaling, distinct from the 2D fp32 cublasSgemm of prior
// cells AND the 4D bf16 cublasGemmStridedBatchedEx of
// problem_083_attention_score_value_matmul and
// problem_032_attention_weights_matmul_with_value_projection.
// The transA=T
// arises because `.transpose(2, 3).to(fp32)` leaves the key
// operand in a non-contiguous transposed-view layout; PyTorch
// forwards the physical [B, H, S, head_dim] fp32 storage to
// cuBLAS with transA=T rather than materializing a [B, H,
// head_dim, S] contiguous copy. Probe runtime-contract fidelity
// is preserved (the cohort-B probe calls cublasSgemmStridedBatched
// directly with the same transA/transB and operand-layout
// contract).
void run(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor attn_weights
) {
    check_bf16_contiguous_cuda(query, "query");
    check_bf16_contiguous_cuda(key, "key");
    check_bf16_contiguous_cuda(attn_weights, "attn_weights");

    constexpr int64_t kNumAttentionHeads = 32;
    constexpr int64_t kNumKvHeads = 32;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNumKvGroups = kNumAttentionHeads / kNumKvHeads;  // 1

    TORCH_CHECK(query.dim() == 4,
                "query must be 4D [B, num_attention_heads, S, head_dim]");
    TORCH_CHECK(key.dim() == 4,
                "key must be 4D [B, num_key_value_heads, S, head_dim]");
    TORCH_CHECK(attn_weights.dim() == 4,
                "attn_weights must be 4D [B, num_attention_heads, S, S]");

    const int64_t B = query.size(0);
    const int64_t S = query.size(2);

    TORCH_CHECK(query.size(1) == kNumAttentionHeads,
                "query.size(1) must equal num_attention_heads=32");
    TORCH_CHECK(query.size(3) == kHeadDim,
                "query.size(3) must equal head_dim=128");

    TORCH_CHECK(key.size(0) == B,
                "key.size(0) must equal query.size(0)");
    TORCH_CHECK(key.size(1) == kNumKvHeads,
                "key.size(1) must equal num_key_value_heads=32");
    TORCH_CHECK(key.size(2) == S,
                "key.size(2) must equal query.size(2) (S)");
    TORCH_CHECK(key.size(3) == kHeadDim,
                "key.size(3) must equal head_dim=128");

    TORCH_CHECK(attn_weights.size(0) == B,
                "attn_weights.size(0) must equal query.size(0)");
    TORCH_CHECK(attn_weights.size(1) == kNumAttentionHeads,
                "attn_weights.size(1) must equal num_attention_heads=32");
    TORCH_CHECK(attn_weights.size(2) == S,
                "attn_weights.size(2) must equal query.size(2) (S)");
    TORCH_CHECK(attn_weights.size(3) == S,
                "attn_weights.size(3) must equal query.size(2) (S) "
                "— self-attention QK^T output");

    // Derive `scaling` from the const `head_dim` (NOT an input).
    // Matches the reference exactly: `scaling = head_dim ** -0.5`.
    const double scaling = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

    // GQA key view chain: expand 32 KV heads to 32 attention heads
    // (n_rep=1). unsqueeze + expand + reshape returns a no-op
    // VIEW — at n_rep=1 the singleton expand axis flattens
    // trivially through reshape with no broadcast or data copy.
    auto key_states = key.unsqueeze(2)
        .expand({B, kNumKvHeads, kNumKvGroups, S, kHeadDim})
        .reshape({B, kNumAttentionHeads, S, kHeadDim});

    // bf16 -> fp32 upcasts (match reference exactly). The
    // .transpose(2, 3) on key_states is a stride-only op; the
    // subsequent .to(fp32) materializes a fp32 tensor that
    // preserves the transposed-view stride pattern (physical
    // [B, H, S, head_dim] fp32 contiguous, logical view
    // [B, H, head_dim, S]).
    auto q_fp32 = query.to(torch::kFloat32);
    auto k_fp32_t = key_states.transpose(2, 3).to(torch::kFloat32);

    // fp32 batched matmul: logical [B, H, S, head_dim] @
    // logical [B, H, head_dim, S] -> [B, H, S, S]. PyTorch's
    // 4D fp32 at::matmul dispatches cublasSgemmStridedBatched
    // (non-Ex fp32 API) with transA=T (because k_fp32_t's
    // physical storage is [B, H, S, head_dim], cuBLAS reads it
    // as transposed) and transB=N (q_fp32 is contiguous).
    auto attn_weights_fp32 = at::matmul(q_fp32, k_fp32_t);

    // Apply derived-constant scaling (broadcast) and downcast to bf16.
    attn_weights.copy_((attn_weights_fp32 * scaling).to(torch::kBFloat16));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp32-inner attention Q@K^T with GQA-style key view (n_rep=1) "
        "and a derived scaling constant. Pure ATen/cuBLAS dispatch "
        "chain: GQA expand + reshape on key (n_rep=1; no-op view) + "
        "bf16->fp32 upcast for query and (transposed) key_states + "
        "one 4D at::matmul dispatching cublasSgemmStridedBatched "
        "(legacy non-Ex fp32 API) with transA=T transB=N + "
        "derived-constant multiply by `scaling = head_dim ** -0.5` "
        "(computed in-kernel from the const head_dim) + bf16 "
        "downcast via .copy_() into pre-allocated DPS output. "
        "num_attention_heads=32, num_key_value_heads=32, head_dim=128, "
        "num_key_value_groups=1 pinned. Destination-passing style.");
}
