#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Complete MoE expert parallel execution with weighted aggregation across the
// selected expert set with externally supplied routing. Each expert processes
// its assigned tokens through a gated MLP (gate_proj, up_proj, SiLU, multiply,
// down_proj),
// then results are weighted by routing weights and scatter-added back to
// original token positions. Reproduces the reference run() op-for-op via pure
// ATen / cuBLAS dispatch -- no custom device kernels.
//
// Dimensions from the selected problem definition:
//   hidden_size         = 8192
//   intermediate_size   = 24576
//   num_experts         = selected definition (projection-weight dim 0)
//   num_experts_per_tok = 2
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states     : [batch_size, seq_len, hidden_size]                 bfloat16
//     routing_weights   : [batch_seq_len, num_experts_per_tok]               float32
//     selected_experts  : [batch_seq_len, num_experts_per_tok]               int64
//     gate_proj_weights : [num_experts, intermediate_size, hidden_size]      bfloat16
//     up_proj_weights   : [num_experts, intermediate_size, hidden_size]      bfloat16
//     down_proj_weights : [num_experts, hidden_size, intermediate_size]      bfloat16
//   output (bfloat16, DPS pre-alloc, written in place):
//     final_hidden_states : [batch_size, seq_len, hidden_size]
//
// Math the reference computes:
//   hidden_states_flat = hidden_states.view(-1, hidden_dim)
//   expert_mask = one_hot(selected_experts, num_experts).permute(2, 1, 0)
//                 -> [num_experts, num_experts_per_tok, batch_seq_len]
//   final_hidden_states = zeros([batch_seq_len, hidden_dim], dtype=bfloat16)
//   per expert e in range(num_experts):
//     idx, top_x = where(expert_mask[e])          (slot index, token index)
//     skip e when no token routes to it (numel guard)
//     current_state = hidden_states_flat[top_x].to(float32)
//     gate_out  = current_state @ gate_w.t().to(float32)
//     up_out    = current_state @ up_w.t().to(float32)
//     gate_activated = silu(gate_out)                            (fp32)
//     intermediate   = gate_activated * up_out
//     expert_output  = intermediate @ down_w.t().to(float32)
//     token_weights  = routing_weights[top_x, idx, None]         (fp32)
//     weighted       = expert_output * token_weights
//     final.index_add_(0, top_x, weighted.to(bfloat16))
//   final_hidden_states = final.reshape(batch_size, seq_len, hidden_dim)
//
// The expert MLP runs entirely in fp32: the gathered tokens and every
// projection weight are upcast to fp32 before the matmuls, the SiLU runs in
// fp32, and the per-(token, slot) routing weight stays fp32 (it is supplied as
// fp32). The accumulator is bf16 (the reference initializes
// final_hidden_states as a bfloat16 zeros buffer); the only narrowing is the
// per-expert .to(bfloat16) on each weighted contribution right before it is
// scatter-added into the bf16 accumulator -- matching the reference's single
// cast point per expert. The one_hot / permute(2, 1, 0) / where construction
// reproduces the reference's exact (idx, top_x) ordering that drives
// index_add_. The first dim of hidden_states is flattened over
// batch_size * seq_len for the per-token gather, and the bf16 accumulator is
// reshaped back to the input's 3D layout for the output. The host loop runs
// over the fixed expert count read from gate_proj_weights.size(0) (the
// reference iterates every expert and skips the empty ones); it issues work
// only on the current PyTorch stream.

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
    const torch::Tensor& gate_proj_weights,
    const torch::Tensor& up_proj_weights,
    const torch::Tensor& down_proj_weights,
    torch::Tensor final_hidden_states
) {
    check_dtype_cuda(hidden_states, torch::kBFloat16, "hidden_states");
    check_dtype_cuda(routing_weights, torch::kFloat32, "routing_weights");
    check_dtype_cuda(selected_experts, torch::kInt64, "selected_experts");
    check_dtype_cuda(gate_proj_weights, torch::kBFloat16, "gate_proj_weights");
    check_dtype_cuda(up_proj_weights, torch::kBFloat16, "up_proj_weights");
    check_dtype_cuda(down_proj_weights, torch::kBFloat16, "down_proj_weights");
    check_dtype_cuda(final_hidden_states, torch::kBFloat16, "final_hidden_states");

    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be 3D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(routing_weights.dim() == 2, "routing_weights must be 2D, got ", routing_weights.dim(), "D");
    TORCH_CHECK(selected_experts.dim() == 2, "selected_experts must be 2D, got ", selected_experts.dim(), "D");
    TORCH_CHECK(gate_proj_weights.dim() == 3, "gate_proj_weights must be 3D, got ", gate_proj_weights.dim(), "D");
    TORCH_CHECK(up_proj_weights.dim() == 3, "up_proj_weights must be 3D, got ", up_proj_weights.dim(), "D");
    TORCH_CHECK(down_proj_weights.dim() == 3, "down_proj_weights must be 3D, got ", down_proj_weights.dim(), "D");
    TORCH_CHECK(final_hidden_states.dim() == 3, "final_hidden_states must be 3D, got ", final_hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t hidden_dim = hidden_states.size(2);
    const int64_t num_experts = gate_proj_weights.size(0);

    // Flatten tokens over batch_size * seq_len for the per-token gather.
    auto hidden_states_flat = hidden_states.view({-1, hidden_dim});

    // bf16 accumulator initialized to zero (reference: final_hidden_states =
    // zeros(..., dtype=bfloat16)).
    auto accum = at::zeros({batch_size * seq_len, hidden_dim},
                           hidden_states.options());

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
        auto gate_w = gate_proj_weights[expert_idx];   // [intermediate, hidden]
        auto up_w = up_proj_weights[expert_idx];       // [intermediate, hidden]
        auto down_w = down_proj_weights[expert_idx];   // [hidden, intermediate]

        // Gather this expert's tokens, upcast to fp32 for the expert MLP.
        auto current_state =
            hidden_states_flat.index_select(/*dim=*/0, top_x).to(torch::kFloat32);

        // Gate / up projections in fp32, then the fp32 SiLU SwiGLU gate.
        auto gate_out = at::matmul(current_state, gate_w.t().to(torch::kFloat32));
        auto up_out = at::matmul(current_state, up_w.t().to(torch::kFloat32));
        auto gate_activated = at::silu(gate_out);
        auto intermediate = gate_activated * up_out;

        // Down projection.
        auto expert_output = at::matmul(intermediate, down_w.t().to(torch::kFloat32));

        // Weight by the per-(token, slot) fp32 routing weight, then narrow to
        // bf16 and accumulate into the bf16 buffer.
        auto token_weights = routing_weights.index({top_x, idx}).unsqueeze(-1);
        auto weighted_output = expert_output * token_weights;
        accum.index_add_(
            /*dim=*/0, top_x, weighted_output.to(torch::kBFloat16));
    }

    // Reshape the bf16 accumulator back to the input's 3D layout.
    auto final_3d = accum.reshape({batch_size, seq_len, hidden_dim});
    TORCH_CHECK(final_hidden_states.sizes() == final_3d.sizes(),
                "final_hidden_states shape ", final_hidden_states.sizes(),
                " does not match computed ", final_3d.sizes());
    final_hidden_states.copy_(final_3d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "MoE expert parallel execution with weighted aggregation (expert "
        "count derived from the projection tensors, externally supplied "
        "routing). Pure ATen dispatch: per-expert host loop over the fixed "
        "expert count gathers each expert's tokens "
        "(one_hot / permute / where / index_select), runs gate+SiLU / up / "
        "SwiGLU / down matmuls in fp32 (tokens and weights upcast), weights by "
        "the fp32 per-(token, slot) routing weight, casts to bfloat16 and "
        "scatter-adds into a bf16 accumulator via index_add_, reproduced "
        "op-for-op from the reference. The bf16 accumulator is reshaped to the "
        "input's 3D layout for the output (destination-passing style).");
}
