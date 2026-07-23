#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32,
                name, " must be float32, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for cross-attention QKV
// projection backward with GQA repeat. Per definition.json:
//   grad_query_states          : [B, num_heads=16, S_dec, head_dim=256]    fp32
//   grad_key_states            : [B, num_heads=16, S_enc, head_dim=256]    fp32
//   grad_value_states          : [B, num_heads=16, S_enc, head_dim=256]    fp32
//   decoder_hidden_states      : [B, S_dec, hidden_size=1536]              fp32
//   encoder_hidden_states      : [B, S_enc, cross_hidden_size=1024]        fp32
//   q_weight                   : [q_out_dim=4096, hidden_size=1536]        fp32
//   k_weight                   : [kv_out_dim=1024, cross_hidden_size=1024] fp32
//   v_weight                   : [kv_out_dim=1024, cross_hidden_size=1024] fp32
//   grad_decoder_hidden_states : [B, S_dec, hidden_size=1536]              fp32 DPS
//   grad_encoder_hidden_states : [B, S_enc, cross_hidden_size=1024]        fp32 DPS
//   grad_q_weight              : [q_out_dim=4096, hidden_size=1536]        fp32 DPS
//   grad_k_weight              : [kv_out_dim=1024, cross_hidden_size=1024] fp32 DPS
//   grad_v_weight              : [kv_out_dim=1024, cross_hidden_size=1024] fp32 DPS
//
// Algorithm (mirrors the reference exactly):
//   Forward was:
//     q_proj = decoder_hidden @ q_weight.T       [B, S_dec, q_out_dim]
//     k_proj = encoder_hidden @ k_weight.T       [B, S_enc, kv_out_dim]
//     v_proj = encoder_hidden @ v_weight.T       [B, S_enc, kv_out_dim]
//     Q = q_proj.view(B, S_dec, num_heads, head_dim).transpose(1, 2)
//     K = k_proj.view(B, S_enc, num_kv_heads, head_dim).transpose(1, 2)
//     V = v_proj.view(B, S_enc, num_kv_heads, head_dim).transpose(1, 2)
//     K_repeated = K.unsqueeze(2).expand(...,n_rep,...).reshape(B,num_heads,S_enc,head_dim)
//     V_repeated = (same)
//
//   Backward through GQA repeat (K / V only):
//     grad_K_un = grad_key_states.view(B, num_kv_heads, num_kv_groups, S_enc, head_dim)
//                                .sum(dim=2)             # [B, num_kv_heads, S_enc, head_dim]
//     grad_V_un = (same)
//
//   Backward through transpose + view (per Q / K / V):
//     grad_q_proj = grad_query_states.transpose(1, 2).contiguous().view(B, S_dec, q_out_dim)
//     grad_k_proj = grad_K_un.transpose(1, 2).contiguous().view(B, S_enc, kv_out_dim)
//     grad_v_proj = grad_V_un.transpose(1, 2).contiguous().view(B, S_enc, kv_out_dim)
//
//   Linear backward:
//     grad_decoder_hidden_states = grad_q_proj @ q_weight        [B, S_dec, hidden_size]
//     grad_encoder_hidden_states = grad_k_proj @ k_weight +
//                                  grad_v_proj @ v_weight        [B, S_enc, cross_hidden_size]
//     grad_q_weight = grad_q_proj.view(-1, q_out_dim).t() @
//                     decoder_hidden.view(-1, hidden_size)        [q_out_dim, hidden_size]
//     grad_k_weight = grad_k_proj.view(-1, kv_out_dim).t() @
//                     encoder_hidden.view(-1, cross_hidden_size)  [kv_out_dim, cross_hidden_size]
//     grad_v_weight = grad_v_proj.view(-1, kv_out_dim).t() @
//                     encoder_hidden.view(-1, cross_hidden_size)  [kv_out_dim, cross_hidden_size]
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   hidden_size       = 1536
//   cross_hidden_size = 1024
//   num_heads         = 16
//   num_kv_heads      = 4
//   num_kv_groups     = num_heads / num_kv_heads = 4
//   head_dim          = 256
//   q_out_dim         = num_heads    * head_dim = 4096
//   kv_out_dim        = num_kv_heads * head_dim = 1024
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA
// kernels. The six 2D fp32 matmul dispatches route through
// cublasSgemm directly. The grad_input path (OP_N OP_N) lands on
// the verified cublasSgemm_basic_fp32 family. The GQA-repeat
// reverse-sum is implemented via `.view(B, num_kv_heads,
// num_kv_groups, S_enc, head_dim).sum(2)` -- the same shape
// math as the reference but with the int-axis sum overload bound
// unambiguously via positional `int64_t` argument (avoids the
// ambiguous-brace-init Tensor::sum form, which resolves to
// either the int-axis or named-axis overload under current
// Torch headers).
void run(
    torch::Tensor grad_query_states,
    torch::Tensor grad_key_states,
    torch::Tensor grad_value_states,
    torch::Tensor decoder_hidden_states,
    torch::Tensor encoder_hidden_states,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight,
    torch::Tensor grad_decoder_hidden_states,
    torch::Tensor grad_encoder_hidden_states,
    torch::Tensor grad_q_weight,
    torch::Tensor grad_k_weight,
    torch::Tensor grad_v_weight
) {
    check_fp32_contiguous_cuda(grad_query_states, "grad_query_states");
    check_fp32_contiguous_cuda(grad_key_states, "grad_key_states");
    check_fp32_contiguous_cuda(grad_value_states, "grad_value_states");
    check_fp32_contiguous_cuda(decoder_hidden_states, "decoder_hidden_states");
    check_fp32_contiguous_cuda(encoder_hidden_states, "encoder_hidden_states");
    check_fp32_contiguous_cuda(q_weight, "q_weight");
    check_fp32_contiguous_cuda(k_weight, "k_weight");
    check_fp32_contiguous_cuda(v_weight, "v_weight");
    check_fp32_contiguous_cuda(grad_decoder_hidden_states, "grad_decoder_hidden_states");
    check_fp32_contiguous_cuda(grad_encoder_hidden_states, "grad_encoder_hidden_states");
    check_fp32_contiguous_cuda(grad_q_weight, "grad_q_weight");
    check_fp32_contiguous_cuda(grad_k_weight, "grad_k_weight");
    check_fp32_contiguous_cuda(grad_v_weight, "grad_v_weight");

    const c10::cuda::CUDAGuard device_guard(grad_query_states.device());

    constexpr int64_t kHiddenSize = 1536;
    constexpr int64_t kCrossHiddenSize = 1024;
    constexpr int64_t kNumHeads = 16;
    constexpr int64_t kNumKvHeads = 4;
    constexpr int64_t kNumKvGroups = kNumHeads / kNumKvHeads;  // 4
    constexpr int64_t kHeadDim = 256;
    constexpr int64_t kQOutDim = kNumHeads * kHeadDim;     // 4096
    constexpr int64_t kKvOutDim = kNumKvHeads * kHeadDim;  // 1024

    TORCH_CHECK(decoder_hidden_states.dim() == 3 &&
                decoder_hidden_states.size(2) == kHiddenSize,
                "decoder_hidden_states must be [B, S_dec, hidden_size=1536]");
    TORCH_CHECK(encoder_hidden_states.dim() == 3 &&
                encoder_hidden_states.size(2) == kCrossHiddenSize,
                "encoder_hidden_states must be [B, S_enc, cross_hidden_size=1024]");
    const int64_t B = decoder_hidden_states.size(0);
    const int64_t S_dec = decoder_hidden_states.size(1);
    const int64_t S_enc = encoder_hidden_states.size(1);
    TORCH_CHECK(encoder_hidden_states.size(0) == B,
                "encoder_hidden_states.size(0) must equal decoder_hidden_states.size(0)");
    TORCH_CHECK(grad_query_states.dim() == 4 &&
                grad_query_states.size(0) == B &&
                grad_query_states.size(1) == kNumHeads &&
                grad_query_states.size(2) == S_dec &&
                grad_query_states.size(3) == kHeadDim,
                "grad_query_states must be [B, num_heads=16, S_dec, head_dim=256]");
    TORCH_CHECK(grad_key_states.dim() == 4 &&
                grad_key_states.size(0) == B &&
                grad_key_states.size(1) == kNumHeads &&
                grad_key_states.size(2) == S_enc &&
                grad_key_states.size(3) == kHeadDim,
                "grad_key_states must be [B, num_heads=16, S_enc, head_dim=256]");
    TORCH_CHECK(grad_value_states.dim() == 4 &&
                grad_value_states.size(0) == B &&
                grad_value_states.size(1) == kNumHeads &&
                grad_value_states.size(2) == S_enc &&
                grad_value_states.size(3) == kHeadDim,
                "grad_value_states must be [B, num_heads=16, S_enc, head_dim=256]");
    TORCH_CHECK(q_weight.dim() == 2 && q_weight.size(0) == kQOutDim &&
                q_weight.size(1) == kHiddenSize,
                "q_weight must be [q_out_dim=4096, hidden_size=1536]");
    TORCH_CHECK(k_weight.dim() == 2 && k_weight.size(0) == kKvOutDim &&
                k_weight.size(1) == kCrossHiddenSize,
                "k_weight must be [kv_out_dim=1024, cross_hidden_size=1024]");
    TORCH_CHECK(v_weight.dim() == 2 && v_weight.size(0) == kKvOutDim &&
                v_weight.size(1) == kCrossHiddenSize,
                "v_weight must be [kv_out_dim=1024, cross_hidden_size=1024]");
    TORCH_CHECK(grad_decoder_hidden_states.sizes() == decoder_hidden_states.sizes(),
                "grad_decoder_hidden_states shape must match decoder_hidden_states");
    TORCH_CHECK(grad_encoder_hidden_states.sizes() == encoder_hidden_states.sizes(),
                "grad_encoder_hidden_states shape must match encoder_hidden_states");
    TORCH_CHECK(grad_q_weight.sizes() == q_weight.sizes(),
                "grad_q_weight shape must match q_weight");
    TORCH_CHECK(grad_k_weight.sizes() == k_weight.sizes(),
                "grad_k_weight shape must match k_weight");
    TORCH_CHECK(grad_v_weight.sizes() == v_weight.sizes(),
                "grad_v_weight shape must match v_weight");

    // Q gradient path: no GQA reverse needed (grad_query_states
    // is already at num_heads heads). Direct transpose + view
    // back to projection layout.
    auto grad_q_proj = grad_query_states.transpose(1, 2).contiguous()
                          .view({B, S_dec, kQOutDim});

    // K / V gradient path: GQA repeat reverse-sum over the
    // repetition dim, then transpose + view back to projection
    // layout. The `.sum(2)` positional int64_t arg binds the
    // int-axis overload of `Tensor::sum` unambiguously (the
    // brace-init form `.sum({2})` is ambiguous against the
    // named-axis overload under current Torch headers).
    auto grad_K_unrepeated = grad_key_states
        .view({B, kNumKvHeads, kNumKvGroups, S_enc, kHeadDim})
        .sum(/*dim=*/2);
    auto grad_k_proj = grad_K_unrepeated.transpose(1, 2).contiguous()
                          .view({B, S_enc, kKvOutDim});

    auto grad_V_unrepeated = grad_value_states
        .view({B, kNumKvHeads, kNumKvGroups, S_enc, kHeadDim})
        .sum(/*dim=*/2);
    auto grad_v_proj = grad_V_unrepeated.transpose(1, 2).contiguous()
                          .view({B, S_enc, kKvOutDim});

    // Linear backward -- grad_decoder_hidden_states (from Q only).
    grad_decoder_hidden_states.copy_(at::matmul(grad_q_proj, q_weight));

    // Linear backward -- grad_encoder_hidden_states (from K + V,
    // accumulated since both K and V project from the same
    // encoder_hidden_states tensor).
    auto grad_encoder_from_k = at::matmul(grad_k_proj, k_weight);
    auto grad_encoder_from_v = at::matmul(grad_v_proj, v_weight);
    grad_encoder_hidden_states.copy_(grad_encoder_from_k + grad_encoder_from_v);

    // Per-projection weight gradients.
    auto grad_q_proj_2d = grad_q_proj.view({-1, kQOutDim});
    auto grad_k_proj_2d = grad_k_proj.view({-1, kKvOutDim});
    auto grad_v_proj_2d = grad_v_proj.view({-1, kKvOutDim});
    auto decoder_2d = decoder_hidden_states.view({-1, kHiddenSize});
    auto encoder_2d = encoder_hidden_states.view({-1, kCrossHiddenSize});
    grad_q_weight.copy_(at::matmul(grad_q_proj_2d.transpose(0, 1), decoder_2d));
    grad_k_weight.copy_(at::matmul(grad_k_proj_2d.transpose(0, 1), encoder_2d));
    grad_v_weight.copy_(at::matmul(grad_v_proj_2d.transpose(0, 1), encoder_2d));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Cross-attention QKV projection backward with GQA repeat. "
        "Pure ATen / cuBLAS dispatch chain: GQA-repeat reverse-sum "
        "on K/V grads (`.view(B, num_kv_heads, num_kv_groups, S_enc, "
        "head_dim).sum(/*dim=*/2)`) + transpose + contiguous + view "
        "layout flips per Q/K/V; six fp32 2D at::matmul calls "
        "(one grad_input matmul into grad_decoder_hidden_states "
        "via Q, two grad_input matmuls summed into "
        "grad_encoder_hidden_states via K+V, and three grad_weight "
        "matmuls per projection head). All matmuls dispatch "
        "cublasSgemm (legacy fp32 API); the grad_input path "
        "(OP_N OP_N) lands on the verified cublasSgemm_basic_fp32 "
        "family. Const axes: hidden_size=1536, cross_hidden_size=1024, "
        "num_heads=16, num_kv_heads=4, num_kv_groups=4, head_dim=256, "
        "q_out_dim=4096, kv_out_dim=1024. Destination-passing style.");
}
