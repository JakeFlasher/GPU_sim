#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <tuple>

// Backward pass for fused GQA QKV projection with RoPE and Q/K RMSNorm.
//
// Reference (problem_091_fused_gqa_qkv_rope_qk_norm_backward): the analytic
// backward written out by hand as explicit tensor algebra (NO torch.autograd).
// Reproduced op-for-op below. The chain inverted, outermost to innermost, is:
// QKV linear projection -> reshape/transpose into heads -> RoPE -> per-head
// RMSNorm. The reference reverses that order: RMSNorm backward first, then RoPE
// backward, then the reshape/transpose, then the linear backward.
//
//   num_heads     = 32
//   num_kv_heads  = 8
//   head_dim      = 128
//   qkv_size      = 32*128 + 2*8*128 = 6144
//
// PRECISION PROFILE (load-bearing -- the two sub-backwards use DIFFERENT
// dtypes, and mixing them up fails correctness):
//
//   _rms_norm_backward INTERNALLY upcasts every operand to float32
//   (grad_output, x, weight, rstd) and only casts grad_x / grad_weight back to
//   bf16 at the return. So that path is genuinely float32 internally and we
//   mirror it with explicit .to(float32) on entry and .to(bfloat16) on exit.
//
//   _rope_backward does NOT upcast anything -- it runs entirely in bf16
//   (cat/mul/add/sum all on bf16 tensors; torch.sum over bf16 still accumulates
//   in fp32 internally via acc_type, which we match by leaving the sum on bf16
//   tensors). Upcasting this path to fp32 would be STRICTLY MORE PRECISE than
//   the bf16 reference, so the accumulated per-op bf16 rounding would diverge
//   and >1% of elements would straddle the rounding boundary -> INCORRECT
//   on every workload (this exact failure mode hit problem_062 / problem_024).
//
//   The reshape/transpose and the linear-projection backward also stay bf16,
//   with a final .to(bfloat16) on each of the six destination buffers (already
//   bf16, but reproduced for faithfulness to the reference's trailing casts).
//
//   _rms_norm_backward(grad_output, x, weight, rstd):  [all float32 internally]
//       x_normed        = x_f32 * rstd_f32
//       grad_weight     = (grad_output_f32 * x_normed).sum(dim=(0,1,2))  # [head_dim]
//       grad_x_direct   = grad_output_f32 * weight_f32 * rstd_f32
//       grad_rstd       = (grad_output_f32 * weight_f32 * x_f32).sum(-1, keepdim=True)
//       grad_x_from_rstd= grad_rstd * (-rstd_f32^3 * x_f32 / head_dim)
//       grad_x          = (grad_x_direct + grad_x_from_rstd).to(bf16)
//       return grad_x, grad_weight.to(bf16)
//
//   _rope_backward(grad_output, x_rotated, cos, sin):  [all bf16]
//       grad_1 = grad_output[..., :half]; grad_2 = grad_output[..., half:]
//       grad_rotated_inv = cat((grad_2, -grad_1), -1)
//       grad_x = grad_output * cos + grad_rotated_inv * sin
//       xr1 = x_rotated[..., :half]; xr2 = x_rotated[..., half:]
//       x_rotated_inv = cat((-xr2, xr1), -1)
//       x_original = x_rotated * cos + x_rotated_inv * sin
//       grad_cos = (grad_output * x_original).sum(dim=1)              # [bsz,seq,head_dim]
//       xo1 = x_original[..., :half]; xo2 = x_original[..., half:]
//       x_original_rotated = cat((-xo2, xo1), -1)
//       grad_sin = (grad_output * x_original_rotated).sum(dim=1)      # [bsz,seq,head_dim]
//       return grad_x, grad_cos, grad_sin
//
//   run(...):
//       grad_query_pre_norm, grad_q_norm_weight = _rms_norm_backward(grad_query, query_pre_norm, q_norm_weight, q_rstd)
//       grad_key_pre_norm,   grad_k_norm_weight = _rms_norm_backward(grad_key,   key_pre_norm,   k_norm_weight, k_rstd)
//       cos_e = cos.unsqueeze(1); sin_e = sin.unsqueeze(1)   # for both q and k
//       grad_query_pre_rope, grad_cos_q, grad_sin_q = _rope_backward(grad_query_pre_norm, query_pre_norm, cos_e, sin_e)
//       grad_key_pre_rope,   grad_cos_k, grad_sin_k = _rope_backward(grad_key_pre_norm,   key_pre_norm,   cos_e, sin_e)
//       grad_cos = grad_cos_q + grad_cos_k ; grad_sin = grad_sin_q + grad_sin_k
//       grad_query_reshaped = grad_query_pre_rope.transpose(1,2)   # [bsz,seq,num_heads,head_dim]
//       grad_key_reshaped   = grad_key_pre_rope.transpose(1,2)
//       grad_value_reshaped = grad_value.transpose(1,2)
//       grad_query_flat = grad_query_reshaped.reshape(bsz, seq, num_heads*head_dim)
//       grad_key_flat   = grad_key_reshaped.reshape(bsz, seq, num_kv_heads*head_dim)
//       grad_value_flat = grad_value_reshaped.reshape(bsz, seq, num_kv_heads*head_dim)
//       grad_qkv_states = cat([grad_query_flat, grad_key_flat, grad_value_flat], -1)
//       grad_qkv_flat = grad_qkv_states.reshape(-1, qkv_size)
//       hidden_flat   = hidden_states.reshape(-1, hidden_size)
//       grad_hidden_states = linear(grad_qkv_flat, qkv_weight.t()).reshape(bsz, seq, hidden_size)
//       grad_qkv_weight    = matmul(grad_qkv_flat.t(), hidden_flat)
//       return (grad_hidden_states, grad_cos, grad_sin, grad_qkv_weight,
//               grad_q_norm_weight, grad_k_norm_weight).to(bf16)
//
// Per definition.json (insertion order):
//   inputs:
//     grad_query     : [bsz, num_heads=32, seq_len, head_dim=128]    bf16
//     grad_key       : [bsz, num_kv_heads=8, seq_len, head_dim]      bf16
//     grad_value     : [bsz, num_kv_heads=8, seq_len, head_dim]      bf16
//     hidden_states  : [bsz, seq_len, hidden_size=4096]              bf16
//     cos            : [bsz, seq_len, head_dim]                      bf16
//     sin            : [bsz, seq_len, head_dim]                      bf16
//     qkv_weight     : [qkv_size=6144, hidden_size=4096]             bf16
//     q_norm_weight  : [head_dim]                                    bf16
//     k_norm_weight  : [head_dim]                                    bf16
//     query_pre_norm : [bsz, num_heads, seq_len, head_dim]           bf16
//     key_pre_norm   : [bsz, num_kv_heads, seq_len, head_dim]        bf16
//     q_rstd         : [bsz, num_heads, seq_len, rstd_dim=1]         bf16
//     k_rstd         : [bsz, num_kv_heads, seq_len, rstd_dim=1]      bf16
//     eps            : scalar                                        fp32 (by-value double, unused -- rstd is precomputed)
//   outputs (all bf16, DPS pre-alloc, written in place):
//     grad_hidden_states : [bsz, seq_len, hidden_size]
//     grad_cos           : [bsz, seq_len, head_dim]
//     grad_sin           : [bsz, seq_len, head_dim]
//     grad_qkv_weight    : [qkv_size, hidden_size]
//     grad_q_norm_weight : [head_dim]
//     grad_k_norm_weight : [head_dim]
//
// Implementation: pure ATen dispatch -- NO custom CUDA kernels, NO autograd.
// eps is accepted to match the signature but is not consumed: the reference's
// _rms_norm_backward takes the precomputed rstd directly and never recomputes
// the variance, so eps does not enter the backward algebra.

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", t.dtype());
}

// RMSNorm backward, float32 internally exactly as the reference: every operand
// is upcast to float32, and grad_x / grad_weight are cast back to bf16 at the
// return. Returns (grad_x_bf16, grad_weight_bf16).
static std::tuple<torch::Tensor, torch::Tensor> rms_norm_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& rstd) {
    auto grad_output_f = grad_output.to(torch::kFloat32);
    auto x_f = x.to(torch::kFloat32);
    auto weight_f = weight.to(torch::kFloat32);
    auto rstd_f = rstd.to(torch::kFloat32);

    // Normalized input.
    auto x_normed = x_f * rstd_f;

    // Gradient w.r.t. weight: reduce batch, head, and sequence axes (0,1,2),
    // leaving the head_dim axis.
    auto grad_weight = (grad_output_f * x_normed).sum(/*dim=*/{0, 1, 2});

    // Gradient w.r.t. input: direct (weight) path plus the rstd path.
    auto grad_x_direct = grad_output_f * weight_f * rstd_f;
    auto grad_rstd =
        (grad_output_f * weight_f * x_f).sum(/*dim=*/-1, /*keepdim=*/true);

    // d(rstd)/d(x) = -rstd^3 * x / D.
    const int64_t D = x.size(-1);
    auto grad_x_from_rstd = grad_rstd * (-rstd_f.pow(3) * x_f / static_cast<double>(D));

    auto grad_x = grad_x_direct + grad_x_from_rstd;

    return std::make_tuple(grad_x.to(torch::kBFloat16),
                           grad_weight.to(torch::kBFloat16));
}

// RoPE backward, bf16 throughout exactly as the reference (no upcast). cos / sin
// arrive already broadcast over the head axis (unsqueeze(1)). Returns
// (grad_x, grad_cos, grad_sin) where grad_cos / grad_sin have already reduced
// the head axis (sum over dim=1).
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rope_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& x_rotated,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
    const int64_t head_dim = grad_output.size(-1);
    const int64_t half = head_dim / 2;

    // Split grad_output into two halves.
    auto grad_1 = grad_output.slice(/*dim=*/-1, /*start=*/0, /*end=*/half);
    auto grad_2 = grad_output.slice(/*dim=*/-1, /*start=*/half, /*end=*/head_dim);

    // Gradient w.r.t. input x (inverse rotation): cat((grad_2, -grad_1)).
    auto grad_rotated_inv = at::cat({grad_2, -grad_1}, /*dim=*/-1);
    auto grad_x = grad_output * cos + grad_rotated_inv * sin;

    // Recover original x from rotated x: cat((-xr2, xr1)).
    auto xr1 = x_rotated.slice(/*dim=*/-1, /*start=*/0, /*end=*/half);
    auto xr2 = x_rotated.slice(/*dim=*/-1, /*start=*/half, /*end=*/head_dim);
    auto x_rotated_inv = at::cat({-xr2, xr1}, /*dim=*/-1);
    auto x_original = x_rotated * cos + x_rotated_inv * sin;

    // Gradient w.r.t. cos: reduce the head axis (dim=1) the broadcast spans.
    auto grad_cos = (grad_output * x_original).sum(/*dim=*/1);

    // Gradient w.r.t. sin via the rotated original: cat((-xo2, xo1)).
    auto xo1 = x_original.slice(/*dim=*/-1, /*start=*/0, /*end=*/half);
    auto xo2 = x_original.slice(/*dim=*/-1, /*start=*/half, /*end=*/head_dim);
    auto x_original_rotated = at::cat({-xo2, xo1}, /*dim=*/-1);
    auto grad_sin = (grad_output * x_original_rotated).sum(/*dim=*/1);

    return std::make_tuple(grad_x, grad_cos, grad_sin);
}

void run(
    const torch::Tensor& grad_query,
    const torch::Tensor& grad_key,
    const torch::Tensor& grad_value,
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& query_pre_norm,
    const torch::Tensor& key_pre_norm,
    const torch::Tensor& q_rstd,
    const torch::Tensor& k_rstd,
    double eps,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_cos,
    torch::Tensor grad_sin,
    torch::Tensor grad_qkv_weight,
    torch::Tensor grad_q_norm_weight,
    torch::Tensor grad_k_norm_weight
) {
    (void)eps;  // Accepted for signature parity; rstd is precomputed (see header).

    check_bf16_cuda(grad_query, "grad_query");
    check_bf16_cuda(grad_key, "grad_key");
    check_bf16_cuda(grad_value, "grad_value");
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(cos, "cos");
    check_bf16_cuda(sin, "sin");
    check_bf16_cuda(qkv_weight, "qkv_weight");
    check_bf16_cuda(q_norm_weight, "q_norm_weight");
    check_bf16_cuda(k_norm_weight, "k_norm_weight");
    check_bf16_cuda(query_pre_norm, "query_pre_norm");
    check_bf16_cuda(key_pre_norm, "key_pre_norm");
    check_bf16_cuda(q_rstd, "q_rstd");
    check_bf16_cuda(k_rstd, "k_rstd");
    check_bf16_cuda(grad_hidden_states, "grad_hidden_states");
    check_bf16_cuda(grad_cos, "grad_cos");
    check_bf16_cuda(grad_sin, "grad_sin");
    check_bf16_cuda(grad_qkv_weight, "grad_qkv_weight");
    check_bf16_cuda(grad_q_norm_weight, "grad_q_norm_weight");
    check_bf16_cuda(grad_k_norm_weight, "grad_k_norm_weight");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [bsz, seq_len, hidden], got ", hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_query.device());

    constexpr int64_t kNumHeads = 32;
    constexpr int64_t kNumKVHeads = 8;
    constexpr int64_t kHeadDim = 128;
    const int64_t qkv_size = kNumHeads * kHeadDim + 2 * kNumKVHeads * kHeadDim;  // 6144

    const int64_t bsz = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t hidden_size = hidden_states.size(2);

    // RMSNorm backward (float32 internally, returns bf16).
    torch::Tensor grad_query_pre_norm, grad_q_norm_weight_result;
    std::tie(grad_query_pre_norm, grad_q_norm_weight_result) =
        rms_norm_backward(grad_query, query_pre_norm, q_norm_weight, q_rstd);

    torch::Tensor grad_key_pre_norm, grad_k_norm_weight_result;
    std::tie(grad_key_pre_norm, grad_k_norm_weight_result) =
        rms_norm_backward(grad_key, key_pre_norm, k_norm_weight, k_rstd);

    // RoPE backward (bf16). cos / sin broadcast over the head axis via
    // unsqueeze(1); the same expansion is used for both the q and k paths.
    auto cos_expanded = cos.unsqueeze(1);  // [bsz,1,seq_len,head_dim]
    auto sin_expanded = sin.unsqueeze(1);  // [bsz,1,seq_len,head_dim]

    torch::Tensor grad_query_pre_rope, grad_cos_q, grad_sin_q;
    std::tie(grad_query_pre_rope, grad_cos_q, grad_sin_q) =
        rope_backward(grad_query_pre_norm, query_pre_norm, cos_expanded, sin_expanded);

    torch::Tensor grad_key_pre_rope, grad_cos_k, grad_sin_k;
    std::tie(grad_key_pre_rope, grad_cos_k, grad_sin_k) =
        rope_backward(grad_key_pre_norm, key_pre_norm, cos_expanded, sin_expanded);

    // Combine cos / sin gradients from the query and key paths.
    auto grad_cos_total = grad_cos_q + grad_cos_k;
    auto grad_sin_total = grad_sin_q + grad_sin_k;

    // Transpose seq<->head back, then flatten heads.
    auto grad_query_reshaped = grad_query_pre_rope.transpose(1, 2);  // [bsz,seq,num_heads,head_dim]
    auto grad_key_reshaped = grad_key_pre_rope.transpose(1, 2);
    auto grad_value_reshaped = grad_value.transpose(1, 2);

    auto grad_query_flat = grad_query_reshaped.reshape({bsz, seq_len, kNumHeads * kHeadDim});
    auto grad_key_flat = grad_key_reshaped.reshape({bsz, seq_len, kNumKVHeads * kHeadDim});
    auto grad_value_flat = grad_value_reshaped.reshape({bsz, seq_len, kNumKVHeads * kHeadDim});

    // Concatenate in the standard [Q_all | K_all | V_all] layout.
    auto grad_qkv_states = at::cat({grad_query_flat, grad_key_flat, grad_value_flat}, /*dim=*/-1);

    // Linear projection backward (bf16).
    auto grad_qkv_flat = grad_qkv_states.reshape({-1, qkv_size});
    auto hidden_flat = hidden_states.reshape({-1, hidden_size});

    // grad w.r.t. hidden_states: [N, qkv_size] @ [qkv_size, hidden_size].
    auto grad_hidden = at::linear(grad_qkv_flat, qkv_weight.t())
                           .reshape({bsz, seq_len, hidden_size});

    // grad w.r.t. qkv_weight: [qkv_size, N] @ [N, hidden_size].
    auto grad_qkv_weight_result = at::matmul(grad_qkv_flat.t(), hidden_flat);

    // Final casts to bf16 (already bf16; reproduced for faithfulness).
    auto grad_hidden_bf16 = grad_hidden.to(torch::kBFloat16);
    auto grad_cos_bf16 = grad_cos_total.to(torch::kBFloat16);
    auto grad_sin_bf16 = grad_sin_total.to(torch::kBFloat16);
    auto grad_qkv_weight_bf16 = grad_qkv_weight_result.to(torch::kBFloat16);
    auto grad_q_norm_weight_bf16 = grad_q_norm_weight_result.to(torch::kBFloat16);
    auto grad_k_norm_weight_bf16 = grad_k_norm_weight_result.to(torch::kBFloat16);

    TORCH_CHECK(grad_hidden_bf16.sizes() == grad_hidden_states.sizes(),
                "grad_hidden_states result shape ", grad_hidden_bf16.sizes(),
                " does not match pre-allocated ", grad_hidden_states.sizes());
    TORCH_CHECK(grad_cos_bf16.sizes() == grad_cos.sizes(),
                "grad_cos result shape ", grad_cos_bf16.sizes(),
                " does not match pre-allocated ", grad_cos.sizes());
    TORCH_CHECK(grad_sin_bf16.sizes() == grad_sin.sizes(),
                "grad_sin result shape ", grad_sin_bf16.sizes(),
                " does not match pre-allocated ", grad_sin.sizes());
    TORCH_CHECK(grad_qkv_weight_bf16.sizes() == grad_qkv_weight.sizes(),
                "grad_qkv_weight result shape ", grad_qkv_weight_bf16.sizes(),
                " does not match pre-allocated ", grad_qkv_weight.sizes());
    TORCH_CHECK(grad_q_norm_weight_bf16.sizes() == grad_q_norm_weight.sizes(),
                "grad_q_norm_weight result shape ", grad_q_norm_weight_bf16.sizes(),
                " does not match pre-allocated ", grad_q_norm_weight.sizes());
    TORCH_CHECK(grad_k_norm_weight_bf16.sizes() == grad_k_norm_weight.sizes(),
                "grad_k_norm_weight result shape ", grad_k_norm_weight_bf16.sizes(),
                " does not match pre-allocated ", grad_k_norm_weight.sizes());

    grad_hidden_states.copy_(grad_hidden_bf16);
    grad_cos.copy_(grad_cos_bf16);
    grad_sin.copy_(grad_sin_bf16);
    grad_qkv_weight.copy_(grad_qkv_weight_bf16);
    grad_q_norm_weight.copy_(grad_q_norm_weight_bf16);
    grad_k_norm_weight.copy_(grad_k_norm_weight_bf16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for fused GQA QKV projection with RoPE and Q/K RMSNorm. "
        "Pure ATen dispatch, hand-derived analytic algebra (NO autograd): "
        "RMSNorm backward (float32 internally with the direct path plus the "
        "-rstd^3 rstd path, returns bf16) -> RoPE backward (bf16 throughout: "
        "inverse rotation for grad_x, head-axis-reduced grad_cos / grad_sin) -> "
        "transpose + flatten heads -> concat [Q|K|V] -> linear-projection "
        "backward (grad_hidden via grad_qkv @ qkv_weight, grad_qkv_weight via "
        "grad_qkv^T @ hidden). The reference's mixed precision profile is "
        "mirrored exactly -- RMSNorm float32, everything else bf16. "
        "Destination-passing style (6 outputs in definition order).");
}
