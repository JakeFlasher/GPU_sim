#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Backward pass for KV-cache update with RoPE.
//
// Reference (problem_062_kv_cache_update_with_rope_backward): the analytic
// backward written out by hand as explicit tensor algebra (NO torch.autograd).
// Reproduced op-for-op below; final results cast to bfloat16 for the six
// destination buffers.
//
// Per definition.json (insertion order):
//   inputs:
//     grad_key_cache   : [batch_size, num_kv_heads, max_seq_len, head_dim]  bfloat16
//     grad_value_cache : [batch_size, num_kv_heads, max_seq_len, head_dim]  bfloat16
//     key_states       : [batch_size, num_kv_heads, new_seq_len, head_dim]  bfloat16
//     cos              : [batch_size, new_seq_len, head_dim]                bfloat16
//     sin              : [batch_size, new_seq_len, head_dim]                bfloat16
//     cache_position   : [new_seq_len]                                      int64
//   outputs (all bfloat16, DPS pre-alloc, written in place):
//     grad_key_states        : [batch_size, num_kv_heads, new_seq_len, head_dim]
//     grad_value_states      : [batch_size, num_kv_heads, new_seq_len, head_dim]
//     grad_cos               : [batch_size, new_seq_len, head_dim]
//     grad_sin               : [batch_size, new_seq_len, head_dim]
//     grad_key_cache_input   : [batch_size, num_kv_heads, max_seq_len, head_dim]
//     grad_value_cache_input : [batch_size, num_kv_heads, max_seq_len, head_dim]
//
// Forward chain the backward inverts (per new position):
//   k1 = key_states[..., :half_dim]; k2 = key_states[..., half_dim:]
//   rotate_half(k) = cat((-k2, k1), dim=-1)
//   key_rotated = key_states * cos.unsqueeze(1) + rotate_half(key_states) * sin.unsqueeze(1)
//   key_cache[:, :, cache_position]   = key_rotated
//   value_cache[:, :, cache_position] = value_states
//
// Implementation: pure ATen dispatch -- NO custom CUDA kernels. The advanced
// index gather grad_*_cache[:, :, cache_position] becomes index_select on
// dim=2. rotate_half's backward redistributes the sin-path gradient across the
// two halves with the matching sign convention:
//   grad_k1 += grad_rotated[..., half:]      (k1 went into the upper half)
//   grad_k2 += -grad_rotated[..., :half]     (-k2 went into the lower half)
// grad_cos / grad_sin reduce the head axis (sum over dim=1, the num_kv_heads
// axis, matching the cos/sin broadcast via unsqueeze(1)). The cache-input
// grads clone the incoming cache grad and zero the touched positions in place
// via index_fill_ on dim=2 (the faithful op for `cache[:, :, cache_position] = 0`).
// value path is a straight gather (no RoPE). Every op + half-split sign matches
// the reference exactly.

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_key_cache,
    const torch::Tensor& grad_value_cache,
    const torch::Tensor& key_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& cache_position,
    torch::Tensor grad_key_states,
    torch::Tensor grad_value_states,
    torch::Tensor grad_cos,
    torch::Tensor grad_sin,
    torch::Tensor grad_key_cache_input,
    torch::Tensor grad_value_cache_input
) {
    check_bf16_cuda(grad_key_cache, "grad_key_cache");
    check_bf16_cuda(grad_value_cache, "grad_value_cache");
    check_bf16_cuda(key_states, "key_states");
    check_bf16_cuda(cos, "cos");
    check_bf16_cuda(sin, "sin");
    check_bf16_cuda(grad_key_states, "grad_key_states");
    check_bf16_cuda(grad_value_states, "grad_value_states");
    check_bf16_cuda(grad_cos, "grad_cos");
    check_bf16_cuda(grad_sin, "grad_sin");
    check_bf16_cuda(grad_key_cache_input, "grad_key_cache_input");
    check_bf16_cuda(grad_value_cache_input, "grad_value_cache_input");

    TORCH_CHECK(cache_position.is_cuda(), "cache_position must be a CUDA tensor");
    TORCH_CHECK(cache_position.dtype() == torch::kInt64,
                "cache_position must be int64, got ", cache_position.dtype());

    TORCH_CHECK(grad_key_cache.dim() == 4, "grad_key_cache must be 4D, got ", grad_key_cache.dim(), "D");
    TORCH_CHECK(grad_value_cache.dim() == 4, "grad_value_cache must be 4D, got ", grad_value_cache.dim(), "D");
    TORCH_CHECK(key_states.dim() == 4, "key_states must be 4D, got ", key_states.dim(), "D");
    TORCH_CHECK(cos.dim() == 3, "cos must be 3D, got ", cos.dim(), "D");
    TORCH_CHECK(sin.dim() == 3, "sin must be 3D, got ", sin.dim(), "D");
    TORCH_CHECK(cache_position.dim() == 1, "cache_position must be 1D, got ", cache_position.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_key_cache.device());

    const int64_t head_dim = key_states.size(-1);
    const int64_t half_dim = head_dim / 2;

    // PRECISION-PROFILE FIX (R16): the reference run() operates in bfloat16
    // throughout -- there is NO .float() anywhere in it. Computing this analytic
    // backward in fp32 and rounding once at the end is STRICTLY MORE PRECISE than
    // the bf16 reference, so the accumulated per-op bf16 rounding diverges and
    // >1% of elements straddle the rounding boundary -> INCORRECT_NUMERICAL on
    // every workload (see BL-20260527-match-reference-precision-profile-in-port).
    // We therefore mirror the reference exactly in bf16 (no upcast). torch.sum
    // on bf16 still accumulates in fp32 internally (acc_type), matching the
    // reference's own .sum(dim=1).

    // Forward: k1 = key_states[..., :half], k2 = key_states[..., half:],
    //          rotate_half(key_states) = cat((-k2, k1), dim=-1).
    auto k1 = key_states.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim);
    auto k2 = key_states.slice(/*dim=*/-1, /*start=*/half_dim, /*end=*/head_dim);
    auto k_rotated_half = at::cat({-k2, k1}, /*dim=*/-1);

    // cos / sin broadcast over the head axis via unsqueeze(1).
    auto cos_expanded = cos.unsqueeze(1);  // [batch, 1, new_seq_len, head_dim]
    auto sin_expanded = sin.unsqueeze(1);  // [batch, 1, new_seq_len, head_dim]

    // Gather grads from the cache at cache_position (advanced index on dim=2).
    //   grad_key_states_rotated = grad_key_cache[:, :, cache_position]
    //   grad_value_states       = grad_value_cache[:, :, cache_position]
    auto grad_key_states_rotated =
        grad_key_cache.index_select(/*dim=*/2, cache_position);
    auto grad_value_states_bf16 =
        grad_value_cache.index_select(/*dim=*/2, cache_position);

    // Backward through key_rotated = key_states * cos + rotate_half(key) * sin.
    auto grad_from_cos_term = grad_key_states_rotated * cos_expanded;
    auto grad_k_rotated_half = grad_key_states_rotated * sin_expanded;

    auto grad_k_rotated_half_1 =
        grad_k_rotated_half.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim);
    auto grad_k_rotated_half_2 =
        grad_k_rotated_half.slice(/*dim=*/-1, /*start=*/half_dim, /*end=*/head_dim);

    // rotate_half backward: lower half held -k2, upper half held k1.
    auto grad_k2_from_rotate = -grad_k_rotated_half_1;
    auto grad_k1_from_rotate = grad_k_rotated_half_2;

    // Total per-half key gradient = direct cos-term path + rotate_half path.
    auto grad_k1_total =
        grad_from_cos_term.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim) + grad_k1_from_rotate;
    auto grad_k2_total =
        grad_from_cos_term.slice(/*dim=*/-1, /*start=*/half_dim, /*end=*/head_dim) + grad_k2_from_rotate;

    auto grad_key_states_bf16 = at::cat({grad_k1_total, grad_k2_total}, /*dim=*/-1);

    // grad_cos / grad_sin: reduce over the head axis (dim=1) the broadcast spans.
    auto grad_cos_bf16 = (grad_key_states_rotated * key_states).sum(/*dim=*/1);
    auto grad_sin_bf16 = (grad_key_states_rotated * k_rotated_half).sum(/*dim=*/1);

    // Cache-input grads: clone incoming cache grad, then zero the positions
    // that were overwritten by the update (cache[:, :, cache_position] = 0).
    auto grad_key_cache_input_bf16 = grad_key_cache.clone();
    grad_key_cache_input_bf16.index_fill_(/*dim=*/2, cache_position, /*value=*/0.0);

    auto grad_value_cache_input_bf16 = grad_value_cache.clone();
    grad_value_cache_input_bf16.index_fill_(/*dim=*/2, cache_position, /*value=*/0.0);

    TORCH_CHECK(grad_key_states_bf16.sizes() == grad_key_states.sizes(),
                "grad_key_states result shape ", grad_key_states_bf16.sizes(),
                " does not match pre-allocated ", grad_key_states.sizes());
    TORCH_CHECK(grad_value_states_bf16.sizes() == grad_value_states.sizes(),
                "grad_value_states result shape ", grad_value_states_bf16.sizes(),
                " does not match pre-allocated ", grad_value_states.sizes());
    TORCH_CHECK(grad_cos_bf16.sizes() == grad_cos.sizes(),
                "grad_cos result shape ", grad_cos_bf16.sizes(),
                " does not match pre-allocated ", grad_cos.sizes());
    TORCH_CHECK(grad_sin_bf16.sizes() == grad_sin.sizes(),
                "grad_sin result shape ", grad_sin_bf16.sizes(),
                " does not match pre-allocated ", grad_sin.sizes());
    TORCH_CHECK(grad_key_cache_input_bf16.sizes() == grad_key_cache_input.sizes(),
                "grad_key_cache_input result shape ", grad_key_cache_input_bf16.sizes(),
                " does not match pre-allocated ", grad_key_cache_input.sizes());
    TORCH_CHECK(grad_value_cache_input_bf16.sizes() == grad_value_cache_input.sizes(),
                "grad_value_cache_input result shape ", grad_value_cache_input_bf16.sizes(),
                " does not match pre-allocated ", grad_value_cache_input.sizes());

    grad_key_states.copy_(grad_key_states_bf16);
    grad_value_states.copy_(grad_value_states_bf16);
    grad_cos.copy_(grad_cos_bf16);
    grad_sin.copy_(grad_sin_bf16);
    grad_key_cache_input.copy_(grad_key_cache_input_bf16);
    grad_value_cache_input.copy_(grad_value_cache_input_bf16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for KV-cache update with RoPE. Pure ATen dispatch: "
        "analytic grad algebra (gather grads from cache at cache_position via "
        "index_select dim=2, rotate_half backward with the cat((-k2, k1)) sign "
        "convention, grad_cos / grad_sin head-axis reduction sum(dim=1), and "
        "cache-input grads via clone + index_fill_ zeros at cache_position) "
        "reproduced op-for-op from the reference in native bfloat16 (matching "
        "the reference's exact precision profile). Destination-passing style "
        "(6 outputs).");
}
