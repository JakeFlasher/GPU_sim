#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Complete MoE expert computation with externally supplied routing: token
// dispatch to SwiGLU expert MLPs and weighted accumulation of expert outputs.
// Reproduces the reference run() op-for-op via pure ATen / cuBLAS dispatch --
// no custom device kernels.
//
// Pinned axes from the problem definition:
//   hidden_size           = 2048
//   moe_intermediate_size = 768
//   num_experts           = 128
//   num_experts_per_tok   = 8
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states     : [batch_seq_len, hidden_size]                      bfloat16
//     routing_weights   : [batch_seq_len, num_experts_per_tok]              float32
//     selected_experts  : [batch_seq_len, num_experts_per_tok]              int64
//     gate_proj_weights : [num_experts, moe_intermediate_size, hidden_size] bfloat16
//     up_proj_weights   : [num_experts, moe_intermediate_size, hidden_size] bfloat16
//     down_proj_weights : [num_experts, hidden_size, moe_intermediate_size] bfloat16
//   output (bfloat16, DPS pre-alloc, written in place):
//     final_hidden_states : [batch_seq_len, hidden_size]
//
// Math the reference computes, per expert e over the fixed expert count:
//   expert_mask = one_hot(selected_experts, num_experts).permute(2, 1, 0)
//                 -> [num_experts, num_experts_per_tok, batch_seq_len]
//   idx, top_x  = where(expert_mask[e].squeeze(0))   (slot index, token index)
//   skip e when no token routes to it (numel guard)
//   current_state = hidden_states[top_x].to(float32)
//   gate_output   = current_state @ gate_w.t().to(float32)
//   gate_activated = gate_output / (1 + exp(-gate_output))     (SiLU, fp32)
//   up_output     = current_state @ up_w.t().to(float32)
//   intermediate  = gate_activated * up_output
//   expert_output = intermediate @ down_w.t().to(float32)
//   weighted      = expert_output * routing_weights[top_x, idx, None]
//   final_hidden_states.index_add_(0, top_x, weighted.to(bfloat16))
//
// The expert MLP runs entirely in fp32: the gathered tokens and every
// projection weight are upcast to fp32 before the matmuls, the SiLU is the
// reference's explicit fp32 sigmoid-gate form, and the per-(token, slot)
// routing weight stays fp32 (it is supplied as fp32). The only narrowing is
// the trailing .to(bfloat16) on each expert's weighted contribution right
// before it is scattered into the bf16 accumulator -- matching the reference's
// single cast point. The one-hot / permute / where construction reproduces the
// reference's exact (idx, top_x) ordering that drives index_add_. The host
// loop runs over the fixed expert count read from gate_proj_weights.size(0)
// (the reference filters to hit experts first, but iterating every expert and
// skipping the empty ones yields the identical set of accumulations); it
// issues work only on the current PyTorch stream.

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

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(routing_weights.dim() == 2, "routing_weights must be 2D, got ", routing_weights.dim(), "D");
    TORCH_CHECK(selected_experts.dim() == 2, "selected_experts must be 2D, got ", selected_experts.dim(), "D");
    TORCH_CHECK(gate_proj_weights.dim() == 3, "gate_proj_weights must be 3D, got ", gate_proj_weights.dim(), "D");
    TORCH_CHECK(up_proj_weights.dim() == 3, "up_proj_weights must be 3D, got ", up_proj_weights.dim(), "D");
    TORCH_CHECK(down_proj_weights.dim() == 3, "down_proj_weights must be 3D, got ", down_proj_weights.dim(), "D");
    TORCH_CHECK(final_hidden_states.dim() == 2, "final_hidden_states must be 2D, got ", final_hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t num_experts = gate_proj_weights.size(0);

    // Accumulator initialized to zero (reference: final_hidden_states = zeros).
    auto accum = at::zeros_like(final_hidden_states);

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
            hidden_states.index_select(/*dim=*/0, top_x).to(torch::kFloat32);

        // Gate projection followed by the reference's explicit fp32 SiLU.
        auto gate_output = at::matmul(current_state, gate_w.t().to(torch::kFloat32));
        auto gate_activated = gate_output / (1.0 + at::exp(-gate_output));

        // Up projection and SwiGLU element-wise gate.
        auto up_output = at::matmul(current_state, up_w.t().to(torch::kFloat32));
        auto intermediate = gate_activated * up_output;

        // Down projection.
        auto expert_output = at::matmul(intermediate, down_w.t().to(torch::kFloat32));

        // Weight by the per-(token, slot) fp32 routing weight, then accumulate.
        auto weights = routing_weights.index({top_x, idx}).unsqueeze(-1);
        auto weighted_output = expert_output * weights;
        accum.index_add_(
            /*dim=*/0, top_x, weighted_output.to(torch::kBFloat16));
    }

    TORCH_CHECK(final_hidden_states.sizes() == accum.sizes(),
                "final_hidden_states shape ", final_hidden_states.sizes(),
                " does not match computed ", accum.sizes());
    final_hidden_states.copy_(accum);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "MoE expert computation with weighted accumulation (externally "
        "supplied routing). Pure ATen dispatch: per-expert host loop over the "
        "fixed expert count gathers each expert's tokens (one_hot / permute / "
        "where / index_select), runs gate+SiLU / up / SwiGLU / down matmuls in "
        "fp32 (tokens and weights upcast), weights by the fp32 per-token "
        "routing weight, casts to bfloat16 and accumulates via index_add_, "
        "reproduced op-for-op from the reference. Destination-passing style "
        "(output zero-init then accumulated in place).");
}
