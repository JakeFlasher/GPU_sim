#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Sparse MoE expert computation and weighted aggregation with an added gated
// shared expert. Externally supplied routing (selected_experts + routing
// weights) dispatches tokens to per-expert SwiGLU MLPs and accumulates the
// routed contribution; a dense SwiGLU shared expert with a sigmoid gate is
// then evaluated on every token and added to the routed result. Reproduces the
// reference run() op-for-op via pure ATen / cuBLAS dispatch -- no custom device
// kernels.
//
// Pinned axes from the problem definition:
//   hidden_size                     = 2048
//   moe_intermediate_size           = 512
//   shared_expert_intermediate_size = 512
//   num_experts                     = 512
//   num_experts_per_tok             = 10
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states                  : [batch_seq_len, hidden_size]                      bfloat16
//     routing_weights                : [batch_seq_len, num_experts_per_tok]              bfloat16
//     selected_experts               : [batch_seq_len, num_experts_per_tok]              int64
//     expert_gate_proj_weights       : [num_experts, moe_intermediate_size, hidden_size] bfloat16
//     expert_up_proj_weights         : [num_experts, moe_intermediate_size, hidden_size] bfloat16
//     expert_down_proj_weights       : [num_experts, hidden_size, moe_intermediate_size] bfloat16
//     shared_expert_gate_proj_weight : [shared_expert_intermediate_size, hidden_size]    bfloat16
//     shared_expert_up_proj_weight   : [shared_expert_intermediate_size, hidden_size]    bfloat16
//     shared_expert_down_proj_weight : [hidden_size, shared_expert_intermediate_size]    bfloat16
//     shared_expert_gate_weight      : [1, hidden_size]                                  bfloat16
//   output (bfloat16, DPS pre-alloc, written in place):
//     output : [batch_seq_len, hidden_size]
//
// Routed path the reference computes, per expert e over the fixed expert count:
//   expert_mask = one_hot(selected_experts, num_experts).permute(2, 1, 0)
//                 -> [num_experts, num_experts_per_tok, batch_seq_len]
//   idx, top_x  = where(expert_mask[e])   (slot index, token index)
//   skip e when no token routes to it (numel guard)
//   current_state = hidden_states[top_x]                              (bf16)
//   gate_out  = current_state @ gate_w.t()                           (bf16)
//   up_out    = current_state @ up_w.t()                             (bf16)
//   silu_gate = gate_out / (1 + exp(-gate_out.float())).to(bf16)
//   intermediate  = silu_gate * up_out                               (bf16)
//   expert_output = intermediate @ down_w.t()                        (bf16)
//   weighted      = expert_output * routing_weights[top_x, idx, None] (bf16)
//   final_hidden_states.index_add_(0, top_x, weighted)
//
// Shared path the reference computes (every token):
//   shared_gate      = hidden_states @ shared_gate_proj_w.t()        (bf16)
//   shared_gate_silu = shared_gate / (1 + exp(-shared_gate.float())).to(bf16)
//   shared_up        = hidden_states @ shared_up_proj_w.t()          (bf16)
//   shared_inter     = shared_gate_silu * shared_up                  (bf16)
//   shared_output    = shared_inter @ shared_down_proj_w.t()         (bf16)
//   gate_logits      = hidden_states @ shared_gate_weight.t()        (bf16)
//   gate             = sigmoid(gate_logits.float()).to(bf16)
//   shared_output    = gate * shared_output                          (bf16)
//   output           = final_hidden_states + shared_output           (bf16)
//
// Precision mirrors the reference exactly. The routed expert MLPs, the routing
// weight multiply, and the shared expert MLP all run in bf16, just as the
// reference's matmuls and bf16 routing tensor do -- no fp32 upcast on either
// projection path. Only the two activation gates promote: the SiLU denominator
// (1 + exp(-x)) is evaluated in fp32 and cast back to bf16 (both the routed and
// shared SwiGLU gates), and the shared expert's scalar gate is a fp32 sigmoid
// of fp32 gate logits cast back to bf16. The one-hot / permute / where
// construction reproduces the reference's exact (idx, top_x) ordering that
// drives index_add_. The host loop runs over the fixed expert count read from
// expert_gate_proj_weights.size(0) (the reference filters to hit experts first,
// but iterating every expert and skipping the empty ones yields the identical
// set of accumulations); it issues work only on the current PyTorch stream.

static void check_dtype_cuda(const torch::Tensor& t,
                             torch::ScalarType dt,
                             const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == dt, name, " has wrong dtype, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& routing_weights,
    const torch::Tensor& selected_experts,
    const torch::Tensor& expert_gate_proj_weights,
    const torch::Tensor& expert_up_proj_weights,
    const torch::Tensor& expert_down_proj_weights,
    const torch::Tensor& shared_expert_gate_proj_weight,
    const torch::Tensor& shared_expert_up_proj_weight,
    const torch::Tensor& shared_expert_down_proj_weight,
    const torch::Tensor& shared_expert_gate_weight,
    torch::Tensor output
) {
    check_dtype_cuda(hidden_states, torch::kBFloat16, "hidden_states");
    check_dtype_cuda(routing_weights, torch::kBFloat16, "routing_weights");
    check_dtype_cuda(selected_experts, torch::kInt64, "selected_experts");
    check_dtype_cuda(expert_gate_proj_weights, torch::kBFloat16, "expert_gate_proj_weights");
    check_dtype_cuda(expert_up_proj_weights, torch::kBFloat16, "expert_up_proj_weights");
    check_dtype_cuda(expert_down_proj_weights, torch::kBFloat16, "expert_down_proj_weights");
    check_dtype_cuda(shared_expert_gate_proj_weight, torch::kBFloat16, "shared_expert_gate_proj_weight");
    check_dtype_cuda(shared_expert_up_proj_weight, torch::kBFloat16, "shared_expert_up_proj_weight");
    check_dtype_cuda(shared_expert_down_proj_weight, torch::kBFloat16, "shared_expert_down_proj_weight");
    check_dtype_cuda(shared_expert_gate_weight, torch::kBFloat16, "shared_expert_gate_weight");
    check_dtype_cuda(output, torch::kBFloat16, "output");

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(routing_weights.dim() == 2, "routing_weights must be 2D, got ", routing_weights.dim(), "D");
    TORCH_CHECK(selected_experts.dim() == 2, "selected_experts must be 2D, got ", selected_experts.dim(), "D");
    TORCH_CHECK(expert_gate_proj_weights.dim() == 3, "expert_gate_proj_weights must be 3D, got ", expert_gate_proj_weights.dim(), "D");
    TORCH_CHECK(expert_up_proj_weights.dim() == 3, "expert_up_proj_weights must be 3D, got ", expert_up_proj_weights.dim(), "D");
    TORCH_CHECK(expert_down_proj_weights.dim() == 3, "expert_down_proj_weights must be 3D, got ", expert_down_proj_weights.dim(), "D");
    TORCH_CHECK(shared_expert_gate_proj_weight.dim() == 2, "shared_expert_gate_proj_weight must be 2D, got ", shared_expert_gate_proj_weight.dim(), "D");
    TORCH_CHECK(shared_expert_up_proj_weight.dim() == 2, "shared_expert_up_proj_weight must be 2D, got ", shared_expert_up_proj_weight.dim(), "D");
    TORCH_CHECK(shared_expert_down_proj_weight.dim() == 2, "shared_expert_down_proj_weight must be 2D, got ", shared_expert_down_proj_weight.dim(), "D");
    TORCH_CHECK(shared_expert_gate_weight.dim() == 2, "shared_expert_gate_weight must be 2D, got ", shared_expert_gate_weight.dim(), "D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D, got ", output.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t num_experts = expert_gate_proj_weights.size(0);

    // Routed accumulator initialized to zero (reference: final_hidden_states = zeros).
    auto final_hidden_states = at::zeros_like(hidden_states);

    // expert_mask: [num_experts, num_experts_per_tok, batch_seq_len].
    auto expert_mask =
        at::one_hot(selected_experts, num_experts).permute({2, 1, 0});

    for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        // where on the [num_experts_per_tok, batch_seq_len] slice returns
        // {slot indices, token indices} in row-major order -- matching the
        // reference's (idx, top_x) unpacking.
        auto nz = at::where(expert_mask[expert_idx]);
        auto idx = nz[0];     // slot indices
        auto top_x = nz[1];   // token indices

        if (top_x.numel() == 0) {
            continue;
        }

        // Per-expert projection weights.
        auto gate_w = expert_gate_proj_weights[expert_idx];   // [intermediate, hidden]
        auto up_w = expert_up_proj_weights[expert_idx];       // [intermediate, hidden]
        auto down_w = expert_down_proj_weights[expert_idx];   // [hidden, intermediate]

        // Gather this expert's tokens (bf16, no upcast on the expert path).
        auto current_state = hidden_states.index_select(/*dim=*/0, top_x);

        // Gate / up projections in bf16.
        auto gate_out = at::matmul(current_state, gate_w.t());
        auto up_out = at::matmul(current_state, up_w.t());

        // SiLU in the reference's mixed form: fp32 sigmoid denominator cast
        // back to bf16, then the bf16 gate_out divided by it.
        auto silu_gate = gate_out /
            (1.0 + at::exp(-gate_out.to(torch::kFloat32))).to(torch::kBFloat16);
        auto intermediate = silu_gate * up_out;

        // Down projection.
        auto expert_output = at::matmul(intermediate, down_w.t());

        // Weight by the per-(token, slot) bf16 routing weight, then accumulate.
        auto weights = routing_weights.index({top_x, idx}).unsqueeze(-1);
        auto weighted_output = expert_output * weights;
        final_hidden_states.index_add_(/*dim=*/0, top_x, weighted_output);
    }

    // Shared expert evaluated densely on every token.
    auto shared_gate = at::matmul(hidden_states, shared_expert_gate_proj_weight.t());
    auto shared_gate_silu = shared_gate /
        (1.0 + at::exp(-shared_gate.to(torch::kFloat32))).to(torch::kBFloat16);
    auto shared_up = at::matmul(hidden_states, shared_expert_up_proj_weight.t());
    auto shared_intermediate = shared_gate_silu * shared_up;
    auto shared_expert_output =
        at::matmul(shared_intermediate, shared_expert_down_proj_weight.t());

    // Sigmoid gate on the shared expert: fp32 sigmoid of fp32 gate logits cast
    // back to bf16, then broadcast-multiplied against the shared output.
    auto gate_logits = at::matmul(hidden_states, shared_expert_gate_weight.t());
    auto shared_gate_weight = at::sigmoid(gate_logits.to(torch::kFloat32)).to(torch::kBFloat16);
    shared_expert_output = shared_gate_weight * shared_expert_output;

    // Combine routed experts and shared expert.
    auto result = final_hidden_states + shared_expert_output;

    TORCH_CHECK(output.sizes() == result.sizes(),
                "output shape ", output.sizes(),
                " does not match computed ", result.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "MoE expert weighted aggregation with a gated shared expert. Pure ATen "
        "dispatch: a per-expert host loop over the fixed expert count gathers "
        "each expert's tokens (one_hot / permute / where / index_select), runs "
        "gate / up / down projections and the mixed-precision SiLU SwiGLU gate "
        "in bf16, weights by the bf16 per-token routing weight, and accumulates "
        "via index_add_; a dense SwiGLU shared expert with a fp32-sigmoid gate "
        "is added to every token, reproduced op-for-op from the reference. "
        "Destination-passing style (output written in place).");
}
