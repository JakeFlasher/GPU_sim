#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Parallel execution of routed MoE experts with token-to-expert dispatch and
// weighted aggregation. Each expert is a SwiGLU MLP (gate_proj, up_proj,
// down_proj). Reproduces the reference run() op-for-op via pure ATen / cuBLAS
// dispatch -- no custom device kernels.
//
// Dimensions from the selected problem definition:
//   hidden_size           = 4096
//   moe_intermediate_size = 2048
//   n_routed_experts      = selected definition (projection-weight dim 0)
//   num_experts_per_tok   = 8
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states     : [num_tokens, hidden_size]                          bfloat16
//     topk_indices      : [num_tokens, num_experts_per_tok]                  int64
//     topk_weights      : [num_tokens, num_experts_per_tok]                  bfloat16
//     gate_proj_weights : [n_routed_experts, moe_intermediate_size, hidden]  bfloat16
//     up_proj_weights   : [n_routed_experts, moe_intermediate_size, hidden]  bfloat16
//     down_proj_weights : [n_routed_experts, hidden, moe_intermediate_size]  bfloat16
//   output (bfloat16, DPS pre-alloc, written in place):
//     output : [num_tokens, hidden_size]
//
// Math the reference computes, per expert e over the fixed expert count:
//   expert_mask = one_hot(topk_indices, n_routed_experts).permute(2, 0, 1)
//                 -> [n_routed_experts, num_tokens, num_experts_per_tok]
//   token_indices, weight_indices = where(expert_mask[e])  (token, slot)
//   skip e when no token routes to it (numel guard)
//   expert_input  = hidden_states[token_indices].to(float32)
//   expert_weights = topk_weights[token_indices, weight_indices].to(float32)
//   gate_output   = expert_input @ gate_w.t().to(float32)
//   gate_output   = silu(gate_output)                          (fp32)
//   up_output     = expert_input @ up_w.t().to(float32)
//   intermediate  = gate_output * up_output
//   expert_output = intermediate @ down_w.t().to(float32)
//   weighted      = expert_output * expert_weights[:, None]
//   final.index_add_(0, token_indices, weighted)               (fp32 accum)
//   return final.to(bfloat16)
//
// The expert MLP runs entirely in fp32: the gathered tokens and every
// projection weight are upcast to fp32 before the matmuls, the SiLU runs in
// fp32, and the gathered per-(token, slot) routing weight is upcast to fp32.
// The accumulator is fp32 (the reference initializes final_hidden_states as a
// fp32 zeros buffer and scatter-adds the fp32 weighted contributions into it);
// the only narrowing is the trailing .to(bfloat16) on the accumulator right
// before it is written to the output -- matching the reference's single cast
// point. The one_hot / permute(2, 0, 1) / where construction reproduces the
// reference's exact (token_indices, weight_indices) ordering that drives
// index_add_. The routing weights are gathered by their token/slot indices
// rather than the slot index alone (topk_weights varies per slot), matching
// the reference's topk_weights[token_indices, weight_indices] indexing. The
// host loop runs over the fixed expert count read from gate_proj_weights.size(0)
// (the reference iterates every expert and skips the empty ones); it issues
// work only on the current PyTorch stream.

static void check_dtype_cuda(const torch::Tensor& t,
                             torch::ScalarType dt,
                             const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == dt, name, " has wrong dtype, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_weights,
    const torch::Tensor& gate_proj_weights,
    const torch::Tensor& up_proj_weights,
    const torch::Tensor& down_proj_weights,
    torch::Tensor output
) {
    check_dtype_cuda(hidden_states, torch::kBFloat16, "hidden_states");
    check_dtype_cuda(topk_indices, torch::kInt64, "topk_indices");
    check_dtype_cuda(topk_weights, torch::kBFloat16, "topk_weights");
    check_dtype_cuda(gate_proj_weights, torch::kBFloat16, "gate_proj_weights");
    check_dtype_cuda(up_proj_weights, torch::kBFloat16, "up_proj_weights");
    check_dtype_cuda(down_proj_weights, torch::kBFloat16, "down_proj_weights");
    check_dtype_cuda(output, torch::kBFloat16, "output");

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be 2D, got ", topk_indices.dim(), "D");
    TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be 2D, got ", topk_weights.dim(), "D");
    TORCH_CHECK(gate_proj_weights.dim() == 3, "gate_proj_weights must be 3D, got ", gate_proj_weights.dim(), "D");
    TORCH_CHECK(up_proj_weights.dim() == 3, "up_proj_weights must be 3D, got ", up_proj_weights.dim(), "D");
    TORCH_CHECK(down_proj_weights.dim() == 3, "down_proj_weights must be 3D, got ", down_proj_weights.dim(), "D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D, got ", output.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t n_routed_experts = gate_proj_weights.size(0);

    // fp32 accumulator initialized to zero (reference: final_hidden_states =
    // zeros(..., dtype=float32)).
    auto accum = at::zeros(
        {hidden_states.size(0), hidden_states.size(1)},
        hidden_states.options().dtype(torch::kFloat32));

    // expert_mask: [n_routed_experts, num_tokens, num_experts_per_tok].
    auto expert_mask =
        at::one_hot(topk_indices, n_routed_experts).permute({2, 0, 1});

    for (int64_t expert_idx = 0; expert_idx < n_routed_experts; ++expert_idx) {
        // where on the [num_tokens, num_experts_per_tok] slice returns
        // {token indices, slot indices} in row-major order -- matching the
        // reference's (token_indices, weight_indices) unpacking.
        auto nz = at::where(expert_mask[expert_idx]);
        auto token_indices = nz[0];   // token indices
        auto weight_indices = nz[1];  // slot indices

        if (token_indices.numel() == 0) {
            continue;
        }

        // Per-expert projection weights.
        auto gate_w = gate_proj_weights[expert_idx];   // [intermediate, hidden]
        auto up_w = up_proj_weights[expert_idx];       // [intermediate, hidden]
        auto down_w = down_proj_weights[expert_idx];   // [hidden, intermediate]

        // Gather this expert's tokens, upcast to fp32 for the expert MLP.
        auto expert_input =
            hidden_states.index_select(/*dim=*/0, token_indices).to(torch::kFloat32);

        // Per-(token, slot) routing weight, gathered and upcast to fp32.
        auto expert_weights =
            topk_weights.index({token_indices, weight_indices}).to(torch::kFloat32);

        // Gate projection followed by the fp32 SiLU.
        auto gate_output = at::matmul(expert_input, gate_w.t().to(torch::kFloat32));
        gate_output = at::silu(gate_output);

        // Up projection and SwiGLU element-wise gate.
        auto up_output = at::matmul(expert_input, up_w.t().to(torch::kFloat32));
        auto intermediate = gate_output * up_output;

        // Down projection.
        auto expert_output = at::matmul(intermediate, down_w.t().to(torch::kFloat32));

        // Weight by the per-(token, slot) fp32 routing weight, then accumulate
        // into the fp32 buffer.
        auto weighted_output = expert_output * expert_weights.unsqueeze(-1);
        accum.index_add_(/*dim=*/0, token_indices, weighted_output);
    }

    // Single narrowing point: cast the fp32 accumulator to bfloat16.
    auto output_bf16 = accum.to(torch::kBFloat16);
    TORCH_CHECK(output.sizes() == output_bf16.sizes(),
                "output shape ", output.sizes(),
                " does not match computed ", output_bf16.sizes());
    output.copy_(output_bf16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "MoE expert parallel execution (expert count derived from the "
        "projection tensors, externally supplied top-k routing). Pure ATen "
        "dispatch: per-expert host loop over the fixed expert count gathers "
        "each expert's tokens (one_hot / "
        "permute / where / index_select), runs gate+SiLU / up / SwiGLU / down "
        "matmuls in fp32 (tokens and weights upcast), weights by the fp32 "
        "per-(token, slot) routing weight, and scatter-adds into an fp32 "
        "accumulator via index_add_, reproduced op-for-op from the reference. "
        "The fp32 accumulator is cast to bfloat16 at the single output write "
        "(destination-passing style).");
}
