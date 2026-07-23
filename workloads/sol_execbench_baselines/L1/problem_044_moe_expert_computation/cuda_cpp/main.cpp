#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// MoE expert computation with SwiGLU activation and multi-expert routing.
//
// Reference (problem_044_moe_expert_computation): a per-expert host loop that
// gathers each expert's assigned tokens, runs a gated MLP, weights the result
// by the routing weight, and scatters it back into the output accumulator.
// Reproduced op-for-op below.
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states     : [num_tokens, hidden_size]                         bfloat16
//     expert_ids        : [num_tokens, num_experts_per_tok]                 int64
//     expert_weights    : [num_tokens, num_experts_per_tok]                 bfloat16
//     gate_proj_weights : [num_experts, intermediate_size, hidden_size]     bfloat16
//     up_proj_weights   : [num_experts, intermediate_size, hidden_size]     bfloat16
//     down_proj_weights : [num_experts, hidden_size, intermediate_size]     bfloat16
//   output (bfloat16, DPS pre-alloc, written in place):
//     output            : [num_tokens, hidden_size]
//
// Math the reference computes, per expert e in range(num_experts):
//   expert_mask = one_hot(expert_ids, num_experts).permute(2, 1, 0)
//                 -> [num_experts, num_experts_per_tok, num_tokens]
//   idx, top_x  = where(expert_mask[e])   (slot index, token index)
//   skip e when no token routes to it
//   expert_tokens = hidden_states[top_x]
//   gate_output   = silu(expert_tokens @ gate_proj_weights[e].T)
//   up_output     = expert_tokens @ up_proj_weights[e].T
//   intermediate  = gate_output * up_output                 (SwiGLU gate)
//   expert_output = intermediate @ down_proj_weights[e].T
//   weights       = expert_weights[top_x, idx].unsqueeze(-1)
//   output.index_add_(0, top_x, (expert_output * weights).to(bfloat16))
//
// Implementation: pure ATen dispatch -- NO custom CUDA kernels. The per-expert
// host loop with data-dependent token gather (at::where / index_select /
// index_add_) is legal in the entry point; it issues work only on the current
// PyTorch stream. The matmuls stay in bfloat16 just as the reference's
// torch.matmul calls do (bf16 inputs, no fp32 upcast) and route through the
// same cuBLAS GEMM. The one-hot / permute / where construction reproduces the
// reference's exact (idx, top_x) ordering that drives the index_add_, and the
// trailing .to(bfloat16) cast point matches the reference. The expert loop
// runs over the fixed expert count read from gate_proj_weights.size(0), not a
// data-dependent count.

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& expert_ids,
    const torch::Tensor& expert_weights,
    const torch::Tensor& gate_proj_weights,
    const torch::Tensor& up_proj_weights,
    const torch::Tensor& down_proj_weights,
    torch::Tensor output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(expert_weights, "expert_weights");
    check_bf16_cuda(gate_proj_weights, "gate_proj_weights");
    check_bf16_cuda(up_proj_weights, "up_proj_weights");
    check_bf16_cuda(down_proj_weights, "down_proj_weights");
    check_bf16_cuda(output, "output");

    TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
    TORCH_CHECK(expert_ids.dtype() == torch::kInt64,
                "expert_ids must be int64, got ", expert_ids.dtype());

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(expert_ids.dim() == 2, "expert_ids must be 2D, got ", expert_ids.dim(), "D");
    TORCH_CHECK(expert_weights.dim() == 2, "expert_weights must be 2D, got ", expert_weights.dim(), "D");
    TORCH_CHECK(gate_proj_weights.dim() == 3, "gate_proj_weights must be 3D, got ", gate_proj_weights.dim(), "D");
    TORCH_CHECK(up_proj_weights.dim() == 3, "up_proj_weights must be 3D, got ", up_proj_weights.dim(), "D");
    TORCH_CHECK(down_proj_weights.dim() == 3, "down_proj_weights must be 3D, got ", down_proj_weights.dim(), "D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D, got ", output.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t num_experts = gate_proj_weights.size(0);

    // Initialize the accumulator to zero (reference: output = zeros_like).
    output.zero_();

    // expert_mask: [num_experts, num_experts_per_tok, num_tokens].
    auto expert_mask =
        at::one_hot(expert_ids, num_experts).permute({2, 1, 0});

    // Process each expert's tokens over the fixed expert count.
    for (int64_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        // Find which tokens route to this expert and which slot they took.
        // at::where on a 2D mask returns {row indices (slot), col indices
        // (token)} in row-major order -- matching the reference's
        // (idx, top_x) unpacking.
        auto nz = at::where(expert_mask[expert_id]);
        auto idx = nz[0];     // slot indices
        auto top_x = nz[1];   // token indices

        if (top_x.numel() == 0) {
            continue;
        }

        // Gather this expert's tokens.
        auto expert_tokens = hidden_states.index_select(/*dim=*/0, top_x);

        // Per-expert projection weights.
        auto gate_w = gate_proj_weights[expert_id];   // [intermediate, hidden]
        auto up_w = up_proj_weights[expert_id];       // [intermediate, hidden]
        auto down_w = down_proj_weights[expert_id];   // [hidden, intermediate]

        // Gate projection followed by SiLU.
        auto gate_output = at::matmul(expert_tokens, gate_w.t());
        gate_output = at::silu(gate_output);

        // Up projection.
        auto up_output = at::matmul(expert_tokens, up_w.t());

        // SwiGLU element-wise gate.
        auto intermediate = gate_output * up_output;

        // Down projection.
        auto expert_output = at::matmul(intermediate, down_w.t());

        // Gather the per-(token, slot) routing weight and accumulate.
        auto weights =
            expert_weights.index({top_x, idx}).unsqueeze(-1);
        output.index_add_(
            /*dim=*/0, top_x,
            (expert_output * weights).to(torch::kBFloat16));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "MoE expert computation with SwiGLU activation. Pure ATen dispatch: "
        "per-expert host loop gathers each expert's tokens (one_hot / permute "
        "/ where / index_select), runs gate+SiLU / up / SwiGLU / down "
        "matmuls in bfloat16, weights by the per-token routing weight, and "
        "accumulates into the output via index_add_, reproduced op-for-op "
        "from the reference. Destination-passing style (output zero-init then "
        "accumulated in place).");
}
