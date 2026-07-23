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

// Destination-passing-style entry for the multi-head QKV
// projection backward with RoPE. Per definition.json:
//   grad_query          : [B, num_heads=24, S, head_dim=96]            fp32
//   grad_key            : [B, num_kv_heads=4, S, head_dim=96]          fp32
//   grad_value          : [B, num_kv_heads=4, S, head_dim=96]          fp32
//   hidden_states       : [B, S, hidden_size=1536]                     fp32
//   q_weight            : [q_out_dim=2304, hidden_size=1536]           fp32
//   k_weight            : [kv_out_dim=384, hidden_size=1536]           fp32
//   v_weight            : [kv_out_dim=384, hidden_size=1536]           fp32
//   cos                 : [B, S, head_dim=96]                          fp32
//   sin                 : [B, S, head_dim=96]                          fp32
//   grad_hidden_states  : [B, S, hidden_size=1536]                     fp32 DPS
//   grad_q_weight       : [q_out_dim=2304, hidden_size=1536]           fp32 DPS
//   grad_k_weight       : [kv_out_dim=384, hidden_size=1536]           fp32 DPS
//   grad_v_weight       : [kv_out_dim=384, hidden_size=1536]           fp32 DPS
//
// Algorithm (mirrors the reference exactly):
//   Forward was:
//     q_proj = hidden_states @ q_weight.T     [B, S, q_out_dim]
//     k_proj = hidden_states @ k_weight.T     [B, S, kv_out_dim]
//     v_proj = hidden_states @ v_weight.T     [B, S, kv_out_dim]
//     q = q_proj.view(B, S, num_heads, head_dim).transpose(1, 2)
//     k = k_proj.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
//     v = v_proj.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
//     q_rope = q * cos + rotate_half(q) * sin
//     k_rope = k * cos + rotate_half(k) * sin
//
//   RoPE backward (Q):
//     cos_u = cos.unsqueeze(1)                # [B, 1, S, head_dim]
//     sin_u = sin.unsqueeze(1)
//     grad_q_cos_term = grad_query * cos_u
//     grad_q_sin_term = grad_query * sin_u
//     # reverse_rotate_half([y1, y2]) = [y2, -y1]
//     grad_q_sin_1 = grad_q_sin_term[..., :half_head_dim]
//     grad_q_sin_2 = grad_q_sin_term[..., half_head_dim:]
//     grad_q_sin_reversed = cat([grad_q_sin_2, -grad_q_sin_1], dim=-1)
//     grad_q_unrotated = grad_q_cos_term + grad_q_sin_reversed
//
//   RoPE backward (K) — same chain on grad_key with num_kv_heads.
//
//   Transpose + view back to projection layout (Q):
//     grad_q_proj = grad_q_unrotated.transpose(1, 2).contiguous()
//                                    .view(B, S, num_heads * head_dim)
//
//   Same for grad_k_proj, grad_v_proj (V has no RoPE; skip
//   straight to the transpose / view step).
//
//   Linear backward — accumulate grad_hidden_states from all three:
//     grad_hidden_from_q = grad_q_proj @ q_weight       [B, S, hidden]
//     grad_hidden_from_k = grad_k_proj @ k_weight
//     grad_hidden_from_v = grad_v_proj @ v_weight
//     grad_hidden_states = grad_hidden_from_q + grad_hidden_from_k +
//                          grad_hidden_from_v
//
//   Per-projection weight gradients:
//     grad_q_proj_2d = grad_q_proj.view(-1, num_heads * head_dim)
//     hidden_states_2d = hidden_states.view(-1, hidden_size)
//     grad_q_weight = grad_q_proj_2d.t() @ hidden_states_2d
//     grad_k_weight = grad_k_proj_2d.t() @ hidden_states_2d
//     grad_v_weight = grad_v_proj_2d.t() @ hidden_states_2d
//
// Const axes pinned by definition.json AND enforced on every tensor
// the kernel indexes with that fixed assumption:
//   hidden_size    = 1536
//   num_heads      = 24
//   num_kv_heads   = 4
//   head_dim       = 96
//   half_head_dim  = 48
//   q_out_dim      = num_heads    * head_dim = 2304
//   kv_out_dim     = num_kv_heads * head_dim = 384
//
// Implementation: pure ATen / cuBLAS dispatch — NO custom CUDA
// kernels. All six 2D fp32 matmul dispatches route through
// cublasSgemm directly. The grad_input path (OP_N OP_N) lands on
// the verified cublasSgemm_basic_fp32 family. The RoPE backward
// is elementwise cos / sin arithmetic + a `narrow / cat` reverse-
// rotate-half — no matmul cost beyond the six listed.
void run(
    torch::Tensor grad_query,
    torch::Tensor grad_key,
    torch::Tensor grad_value,
    torch::Tensor hidden_states,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor v_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_q_weight,
    torch::Tensor grad_k_weight,
    torch::Tensor grad_v_weight
) {
    check_fp32_contiguous_cuda(grad_query, "grad_query");
    check_fp32_contiguous_cuda(grad_key, "grad_key");
    check_fp32_contiguous_cuda(grad_value, "grad_value");
    check_fp32_contiguous_cuda(hidden_states, "hidden_states");
    check_fp32_contiguous_cuda(q_weight, "q_weight");
    check_fp32_contiguous_cuda(k_weight, "k_weight");
    check_fp32_contiguous_cuda(v_weight, "v_weight");
    check_fp32_contiguous_cuda(cos, "cos");
    check_fp32_contiguous_cuda(sin, "sin");
    check_fp32_contiguous_cuda(grad_hidden_states, "grad_hidden_states");
    check_fp32_contiguous_cuda(grad_q_weight, "grad_q_weight");
    check_fp32_contiguous_cuda(grad_k_weight, "grad_k_weight");
    check_fp32_contiguous_cuda(grad_v_weight, "grad_v_weight");

    const c10::cuda::CUDAGuard device_guard(grad_query.device());

    constexpr int64_t kHiddenSize = 1536;
    constexpr int64_t kNumHeads = 24;
    constexpr int64_t kNumKvHeads = 4;
    constexpr int64_t kHeadDim = 96;
    constexpr int64_t kHalfHeadDim = kHeadDim / 2;  // 48
    constexpr int64_t kQOutDim = kNumHeads * kHeadDim;     // 2304
    constexpr int64_t kKvOutDim = kNumKvHeads * kHeadDim;  // 384

    TORCH_CHECK(hidden_states.dim() == 3 &&
                hidden_states.size(2) == kHiddenSize,
                "hidden_states must be [B, S, hidden_size=1536]");
    const int64_t B = hidden_states.size(0);
    const int64_t S = hidden_states.size(1);
    TORCH_CHECK(grad_query.dim() == 4 && grad_query.size(0) == B &&
                grad_query.size(1) == kNumHeads && grad_query.size(2) == S &&
                grad_query.size(3) == kHeadDim,
                "grad_query must be [B, num_heads=24, S, head_dim=96]");
    TORCH_CHECK(grad_key.dim() == 4 && grad_key.size(0) == B &&
                grad_key.size(1) == kNumKvHeads && grad_key.size(2) == S &&
                grad_key.size(3) == kHeadDim,
                "grad_key must be [B, num_kv_heads=4, S, head_dim=96]");
    TORCH_CHECK(grad_value.dim() == 4 && grad_value.size(0) == B &&
                grad_value.size(1) == kNumKvHeads && grad_value.size(2) == S &&
                grad_value.size(3) == kHeadDim,
                "grad_value must be [B, num_kv_heads=4, S, head_dim=96]");
    TORCH_CHECK(q_weight.dim() == 2 && q_weight.size(0) == kQOutDim &&
                q_weight.size(1) == kHiddenSize,
                "q_weight must be [q_out_dim=2304, hidden_size=1536]");
    TORCH_CHECK(k_weight.dim() == 2 && k_weight.size(0) == kKvOutDim &&
                k_weight.size(1) == kHiddenSize,
                "k_weight must be [kv_out_dim=384, hidden_size=1536]");
    TORCH_CHECK(v_weight.dim() == 2 && v_weight.size(0) == kKvOutDim &&
                v_weight.size(1) == kHiddenSize,
                "v_weight must be [kv_out_dim=384, hidden_size=1536]");
    TORCH_CHECK(cos.dim() == 3 && cos.size(0) == B && cos.size(1) == S &&
                cos.size(2) == kHeadDim,
                "cos must be [B, S, head_dim=96]");
    TORCH_CHECK(sin.dim() == 3 && sin.size(0) == B && sin.size(1) == S &&
                sin.size(2) == kHeadDim,
                "sin must be [B, S, head_dim=96]");
    TORCH_CHECK(grad_hidden_states.sizes() == hidden_states.sizes(),
                "grad_hidden_states shape must match hidden_states");
    TORCH_CHECK(grad_q_weight.sizes() == q_weight.sizes(),
                "grad_q_weight shape must match q_weight");
    TORCH_CHECK(grad_k_weight.sizes() == k_weight.sizes(),
                "grad_k_weight shape must match k_weight");
    TORCH_CHECK(grad_v_weight.sizes() == v_weight.sizes(),
                "grad_v_weight shape must match v_weight");

    // cos / sin broadcast over the heads axis: unsqueeze to
    // [B, 1, S, head_dim].
    auto cos_u = cos.unsqueeze(1);
    auto sin_u = sin.unsqueeze(1);

    // RoPE backward for Q.
    auto grad_q_cos_term = grad_query * cos_u;
    auto grad_q_sin_term = grad_query * sin_u;
    auto grad_q_sin_1 =
        grad_q_sin_term.narrow(-1, 0, kHalfHeadDim);
    auto grad_q_sin_2 =
        grad_q_sin_term.narrow(-1, kHalfHeadDim, kHalfHeadDim);
    auto grad_q_sin_reversed =
        at::cat({grad_q_sin_2, -grad_q_sin_1}, /*dim=*/-1);
    auto grad_q_unrotated = grad_q_cos_term + grad_q_sin_reversed;

    // RoPE backward for K.
    auto grad_k_cos_term = grad_key * cos_u;
    auto grad_k_sin_term = grad_key * sin_u;
    auto grad_k_sin_1 =
        grad_k_sin_term.narrow(-1, 0, kHalfHeadDim);
    auto grad_k_sin_2 =
        grad_k_sin_term.narrow(-1, kHalfHeadDim, kHalfHeadDim);
    auto grad_k_sin_reversed =
        at::cat({grad_k_sin_2, -grad_k_sin_1}, /*dim=*/-1);
    auto grad_k_unrotated = grad_k_cos_term + grad_k_sin_reversed;

    // Transpose + view back to projection layout for Q / K / V.
    auto grad_q_proj = grad_q_unrotated.transpose(1, 2).contiguous()
                           .view({B, S, kQOutDim});
    auto grad_k_proj = grad_k_unrotated.transpose(1, 2).contiguous()
                           .view({B, S, kKvOutDim});
    auto grad_v_proj = grad_value.transpose(1, 2).contiguous()
                           .view({B, S, kKvOutDim});

    // Linear backward — accumulate grad_hidden_states from the
    // three projection-head contributions.
    auto grad_hidden_from_q = at::matmul(grad_q_proj, q_weight);
    auto grad_hidden_from_k = at::matmul(grad_k_proj, k_weight);
    auto grad_hidden_from_v = at::matmul(grad_v_proj, v_weight);
    grad_hidden_states.copy_(
        grad_hidden_from_q + grad_hidden_from_k + grad_hidden_from_v);

    // Per-projection weight gradients.
    auto grad_q_proj_2d = grad_q_proj.view({-1, kQOutDim});
    auto grad_k_proj_2d = grad_k_proj.view({-1, kKvOutDim});
    auto grad_v_proj_2d = grad_v_proj.view({-1, kKvOutDim});
    auto hidden_states_2d = hidden_states.view({-1, kHiddenSize});
    grad_q_weight.copy_(
        at::matmul(grad_q_proj_2d.transpose(0, 1), hidden_states_2d));
    grad_k_weight.copy_(
        at::matmul(grad_k_proj_2d.transpose(0, 1), hidden_states_2d));
    grad_v_weight.copy_(
        at::matmul(grad_v_proj_2d.transpose(0, 1), hidden_states_2d));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Multi-head QKV projection backward with RoPE. Pure ATen / "
        "cuBLAS dispatch chain: RoPE backward as elementwise cos / sin "
        "arithmetic plus narrow / cat reverse-rotate-half (no matmul); "
        "transpose + contiguous + view layout flip per Q / K / V; six "
        "fp32 2D at::matmul calls (three grad_input contributions "
        "summed into grad_hidden_states, and three grad_weight matmuls "
        "per projection head). All matmuls dispatch cublasSgemm "
        "(legacy fp32 API); the grad_input path (OP_N OP_N) lands on "
        "the verified cublasSgemm_basic_fp32 family. Const axes: "
        "hidden_size=1536, num_heads=24, num_kv_heads=4, head_dim=96, "
        "half_head_dim=48, q_out_dim=2304, kv_out_dim=384. "
        "Destination-passing style.");
}
