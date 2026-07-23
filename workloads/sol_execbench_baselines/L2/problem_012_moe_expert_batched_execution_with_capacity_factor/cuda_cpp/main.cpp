#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>
#include <cmath>

// Capacity-batched MoE expert execution with SwiGLU experts. Reproduces the
// reference run() op-for-op via pure ATen / cuBLAS dispatch -- no custom device
// kernels. The reference is fully vectorized (sort / bincount / cumsum /
// batched matmul / scatter-gather); this port mirrors that vectorized path
// rather than a per-expert host loop.
//
// Dimensions from the selected problem definition:
//   hidden_size           = 6144
//   moe_intermediate_size = 2560
//   num_experts           = selected definition (expert-weight dim 0)
//   num_experts_per_tok   = 8   (K)
//   capacity factor       = 1.25 (fixed)
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states       : [num_tokens, hidden_size]                          bfloat16
//     selected_experts    : [num_tokens, num_experts_per_tok]                  int64
//     routing_weights     : [num_tokens, num_experts_per_tok]                  bfloat16
//     expert_gate_weights : [num_experts, hidden_size, moe_intermediate_size]  bfloat16
//     expert_up_weights   : [num_experts, hidden_size, moe_intermediate_size]  bfloat16
//     expert_down_weights : [num_experts, moe_intermediate_size, hidden_size]  bfloat16
//   output (bfloat16, DPS pre-alloc, written in place):
//     output : [num_tokens, hidden_size]
//
// Math the reference computes:
//   capacity = max(int((num_tokens * K / num_experts) * 1.25), 1)   (host scalar)
//   flatten the (token, expert) routing assignments into 1D:
//     flat_experts   = selected_experts.reshape(-1)
//     flat_weights   = routing_weights.reshape(-1)
//     flat_token_ids = arange(num_tokens).repeat_interleave(K)
//   stable-sort by expert id; carry the weights and token ids along:
//     sorted_experts, sorted_indices = flat_experts.sort(stable=True)
//     sorted_weights   = flat_weights[sorted_indices]
//     sorted_token_ids = flat_token_ids[sorted_indices]
//   per-expert group offsets (after the stable sort, each expert's slots are
//   contiguous), and the within-group position of every slot:
//     counts      = bincount(sorted_experts, minlength=num_experts)
//     starts      = exclusive-cumsum(counts)
//     within_pos  = arange(len) - starts[sorted_experts]
//   admit the first `capacity` slots of each expert (the capacity constraint):
//     valid = within_pos < capacity
//     v_exp / v_pos / v_tok / v_wt = the four sorted tensors masked by `valid`
//   scatter the admitted tokens into a zero-padded [num_experts, capacity,
//   hidden] batch, run the three batched expert matmuls (gate / up / down) with
//   a SwiGLU gate, gather the admitted outputs back, weight by the routing
//   weight, and scatter-add into the per-token result:
//     expert_inputs[v_exp, v_pos] = hidden_states[v_tok]
//     gate_out  = bmm(expert_inputs, expert_gate_weights)
//     up_out    = bmm(expert_inputs, expert_up_weights)
//     activated = silu(gate_out) * up_out
//     expert_outputs = bmm(activated, expert_down_weights)
//     valid_out   = expert_outputs[v_exp, v_pos]
//     weighted    = v_wt[:, None] * valid_out
//     result.index_add_(0, v_tok, weighted)
//
// Precision path: the reference performs every matmul and the SwiGLU activation
// directly in bfloat16 (the batched matmuls accumulate in fp32 internally per
// cuBLAS, but inputs / outputs stay bf16) -- there is NO explicit fp32 upcast
// anywhere in this reference, so this port keeps the entire expert MLP in
// bfloat16 to match the reference casts exactly. The routing weights are
// supplied in bf16 and stay bf16 through the weighting. capacity is the only
// host-resident scalar; it is computed from num_tokens with the reference's
// exact int() truncation and the fixed 1.25 factor.

static void check_dtype_cuda(const torch::Tensor& t,
                             torch::ScalarType dt,
                             const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == dt, name, " has wrong dtype, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& selected_experts,
    const torch::Tensor& routing_weights,
    const torch::Tensor& expert_gate_weights,
    const torch::Tensor& expert_up_weights,
    const torch::Tensor& expert_down_weights,
    torch::Tensor output
) {
    check_dtype_cuda(hidden_states, torch::kBFloat16, "hidden_states");
    check_dtype_cuda(selected_experts, torch::kInt64, "selected_experts");
    check_dtype_cuda(routing_weights, torch::kBFloat16, "routing_weights");
    check_dtype_cuda(expert_gate_weights, torch::kBFloat16, "expert_gate_weights");
    check_dtype_cuda(expert_up_weights, torch::kBFloat16, "expert_up_weights");
    check_dtype_cuda(expert_down_weights, torch::kBFloat16, "expert_down_weights");
    check_dtype_cuda(output, torch::kBFloat16, "output");

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(selected_experts.dim() == 2, "selected_experts must be 2D, got ", selected_experts.dim(), "D");
    TORCH_CHECK(routing_weights.dim() == 2, "routing_weights must be 2D, got ", routing_weights.dim(), "D");
    TORCH_CHECK(expert_gate_weights.dim() == 3, "expert_gate_weights must be 3D, got ", expert_gate_weights.dim(), "D");
    TORCH_CHECK(expert_up_weights.dim() == 3, "expert_up_weights must be 3D, got ", expert_up_weights.dim(), "D");
    TORCH_CHECK(expert_down_weights.dim() == 3, "expert_down_weights must be 3D, got ", expert_down_weights.dim(), "D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D, got ", output.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t num_tokens = hidden_states.size(0);
    const int64_t hidden_size = hidden_states.size(1);
    const int64_t num_experts = expert_gate_weights.size(0);
    const int64_t num_experts_per_tok = selected_experts.size(1);

    // Host-side capacity scalar: reference int() truncates toward zero.
    const int64_t capacity = std::max<int64_t>(
        static_cast<int64_t>(
            (static_cast<double>(num_tokens) * num_experts_per_tok / num_experts) * 1.25),
        1);

    const auto idx_opts =
        torch::TensorOptions().dtype(torch::kInt64).device(hidden_states.device());

    // Flatten the (token, expert) assignments into 1D.
    auto flat_experts = selected_experts.reshape({-1});
    auto flat_weights = routing_weights.reshape({-1});
    auto flat_token_ids =
        at::arange(num_tokens, idx_opts).repeat_interleave(num_experts_per_tok);

    // Stable sort by expert id; carry weights and token ids along.
    auto sorted = at::sort(flat_experts, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    auto sorted_experts = std::get<0>(sorted);
    auto sorted_indices = std::get<1>(sorted);
    auto sorted_weights = flat_weights.index_select(0, sorted_indices);
    auto sorted_token_ids = flat_token_ids.index_select(0, sorted_indices);

    // Per-expert group offsets: each expert's slots are contiguous after the
    // stable sort, so the within-group position is the global sorted index
    // minus the start of that expert's group.
    auto counts = at::bincount(sorted_experts, /*weights=*/{}, /*minlength=*/num_experts);
    auto starts = at::zeros({num_experts}, idx_opts);
    starts.slice(0, 1, num_experts)
        .copy_(counts.slice(0, 0, num_experts - 1).cumsum(0));

    const int64_t total_slots = sorted_experts.size(0);
    auto within_pos =
        at::arange(total_slots, idx_opts) - starts.index_select(0, sorted_experts);

    // Capacity constraint: admit the first `capacity` slots of each expert.
    auto valid = within_pos < capacity;
    auto v_exp = sorted_experts.masked_select(valid);
    auto v_pos = within_pos.masked_select(valid);
    auto v_tok = sorted_token_ids.masked_select(valid);
    auto v_wt = sorted_weights.masked_select(valid);

    // Scatter admitted tokens into a zero-padded [num_experts, capacity, hidden]
    // batch via advanced indexing.
    auto expert_inputs = at::zeros(
        {num_experts, capacity, hidden_size},
        hidden_states.options());
    expert_inputs.index_put_(
        {v_exp, v_pos},
        hidden_states.index_select(0, v_tok));

    // Batched expert forward pass: gate / up projections, SwiGLU gate, down
    // projection -- all in bfloat16 to match the reference.
    auto gate_out = at::bmm(expert_inputs, expert_gate_weights);
    auto up_out = at::bmm(expert_inputs, expert_up_weights);
    auto activated = at::silu(gate_out).mul_(up_out);
    auto expert_outputs = at::bmm(activated, expert_down_weights);

    // Gather admitted outputs, weight by the routing weight, scatter-add into
    // the per-token result.
    auto valid_out = expert_outputs.index({v_exp, v_pos});
    auto weighted_out = v_wt.unsqueeze(1) * valid_out;

    auto result = at::zeros({num_tokens, hidden_size}, hidden_states.options());
    result.index_add_(0, v_tok, weighted_out);

    TORCH_CHECK(output.sizes() == result.sizes(),
                "output shape ", output.sizes(),
                " does not match computed ", result.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Capacity-batched MoE expert execution with SwiGLU experts. Pure ATen "
        "dispatch reproducing the vectorized reference: flatten and stable-sort "
        "the (token, expert) routing assignments, derive per-expert group "
        "offsets via bincount / exclusive-cumsum, admit the first `capacity` "
        "slots per expert (capacity computed host-side from num_tokens with the "
        "fixed 1.25 factor), scatter admitted tokens into a padded "
        "[num_experts, capacity, hidden] batch, run the three batched matmuls "
        "(gate / up / down) with a SwiGLU gate entirely in bfloat16, gather the "
        "admitted outputs, weight by the bf16 routing weight, and scatter-add "
        "into the per-token output via index_add_, reproduced op-for-op from "
        "the reference. Destination-passing style (output written in place).");
}
