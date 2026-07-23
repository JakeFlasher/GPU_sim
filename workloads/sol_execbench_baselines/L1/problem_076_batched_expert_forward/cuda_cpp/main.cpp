#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Batched MoE expert forward pass (gpt-oss-120b inference path).
//
// Reference (problem_076_batched_expert_forward): every expert is evaluated in
// parallel over the full token set. The hidden states are replicated across
// experts, a batched gate_up projection runs, the interleaved gate / up halves
// are split out and clamped, a custom gated GLU activation fires, a batched
// down projection follows, and the per-expert results are weighted by the
// routing weights and summed. Reproduced op-for-op below.
//
// Per definition.json (insertion order):
//   inputs (all float32 except the scalar alpha / limit):
//     hidden_states     : [batch_size, seq_len, hidden_size]
//     routing_weights   : [tokens, num_experts]   (tokens = batch_size * seq_len)
//     gate_up_proj      : [num_experts, hidden_size, gate_up_dim]  (gate_up_dim = 2 * expert_dim)
//     gate_up_proj_bias : [num_experts, gate_up_dim]
//     down_proj         : [num_experts, expert_dim, hidden_size]
//     down_proj_bias    : [num_experts, hidden_size]
//     alpha             : scalar (passed as double)
//     limit             : scalar (passed as double)
//   output (float32, DPS pre-alloc, written in place):
//     output            : [batch_size, seq_len, hidden_size]
//
// The gate / up halves are interleaved along the last axis of the projection:
// gate takes the even positions (start 0, step 2), up the odd positions
// (start 1, step 2). The clamp is asymmetric -- gate is bounded above by the
// limit only, up is bounded both below by -limit and above by limit. The GLU
// is gate * sigmoid(gate * alpha) scaled by (up + 1), so alpha multiplies the
// gate inside the sigmoid and limit bounds both halves before activation.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// at::bmm calls route through the same batched cuBLAS GEMM torch.bmm issues,
// every tensor stays float32 throughout (matching the reference's profile, no
// casts), and the strided slices / clamp bounds / activation form / routing
// reduction order all match the reference. Output is written in place into the
// pre-allocated destination (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& routing_weights,
    const torch::Tensor& gate_up_proj,
    const torch::Tensor& gate_up_proj_bias,
    const torch::Tensor& down_proj,
    const torch::Tensor& down_proj_bias,
    double alpha,
    double limit,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(routing_weights, "routing_weights");
    check_fp32_cuda(gate_up_proj, "gate_up_proj");
    check_fp32_cuda(gate_up_proj_bias, "gate_up_proj_bias");
    check_fp32_cuda(down_proj, "down_proj");
    check_fp32_cuda(down_proj_bias, "down_proj_bias");
    check_fp32_cuda(output, "output");
    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, hidden], got ", hidden_states.dim(), "D");
    TORCH_CHECK(gate_up_proj.dim() == 3,
                "gate_up_proj must be 3D [E, hidden, gate_up_dim], got ", gate_up_proj.dim(), "D");
    TORCH_CHECK(down_proj.dim() == 3,
                "down_proj must be 3D [E, expert_dim, hidden], got ", down_proj.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t hidden_size = hidden_states.size(2);
    const int64_t num_experts = gate_up_proj.size(0);

    // Flatten the batch and sequence axes into one token axis.
    auto hidden_flat = hidden_states.reshape({-1, hidden_size});

    // Replicate the tokens across every expert, then split the replicas back
    // out into the expert batch axis: (num_experts, tokens, hidden).
    auto hidden_repeated = hidden_flat.repeat({num_experts, 1});
    auto hidden_batched = hidden_repeated.view({num_experts, -1, hidden_size});

    // Batched gate_up projection with a broadcast bias add over the token axis.
    auto gate_up = at::bmm(hidden_batched, gate_up_proj);
    gate_up = gate_up + gate_up_proj_bias.unsqueeze(1);

    // Split the interleaved halves: gate from the even lane, up from the odd
    // lane, each stepping by two along the last axis.
    const int64_t last_dim = gate_up.dim() - 1;
    const int64_t end = gate_up.size(last_dim);
    auto gate = at::slice(gate_up, last_dim, /*start=*/0, /*end=*/end, /*step=*/2);
    auto up = at::slice(gate_up, last_dim, /*start=*/1, /*end=*/end, /*step=*/2);

    // Clamp before activation: gate is bounded above only, up on both sides.
    gate = at::clamp(gate, /*min=*/c10::nullopt, /*max=*/limit);
    up = at::clamp(up, /*min=*/-limit, /*max=*/limit);

    // Custom gated GLU: gate * sigmoid(gate * alpha), scaled by (up + 1).
    auto glu = gate * at::sigmoid(gate * alpha);
    auto gated_output = (up + 1) * glu;

    // Batched down projection with a broadcast bias add over the token axis.
    auto expert_outputs = at::bmm(gated_output, down_proj);
    expert_outputs = expert_outputs + down_proj_bias.unsqueeze(1);

    // Restore the (num_experts, batch, seq, hidden) layout for routing.
    expert_outputs = expert_outputs.view({num_experts, batch_size, seq_len, hidden_size});

    // Bring the routing weights to (num_experts, batch, seq, 1) so they
    // broadcast over the hidden axis.
    auto routing_weights_reshaped =
        routing_weights.transpose(0, 1).view({num_experts, batch_size, seq_len, 1});

    // Weight each expert's output and sum across the expert axis.
    auto result = (expert_outputs * routing_weights_reshaped).sum(/*dim=*/0);

    TORCH_CHECK(result.sizes() == output.sizes(),
                "expert forward output shape ", result.sizes(),
                " does not match pre-allocated output ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Batched MoE expert forward pass via at::bmm gate_up / down "
          "projections + interleaved gate / up split + asymmetric clamp + "
          "gated GLU + routing-weighted sum (pure ATen / cuBLAS dispatch); "
          "fp32; destination-passing style.");
}
