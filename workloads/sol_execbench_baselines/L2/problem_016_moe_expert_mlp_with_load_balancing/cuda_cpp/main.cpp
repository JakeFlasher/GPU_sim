#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Complete MoE expert computation with internal gate routing and load
// balancing: linear gate -> fp32 softmax -> top-8 selection -> SwiGLU expert
// MLPs -> weighted aggregation, plus the raw router logits for the
// load-balancing loss. Reproduces the reference run() op-for-op via pure ATen
// / cuBLAS dispatch -- no custom device kernels.
//
// Pinned axes from the problem definition:
//   hidden_size           = 4096
//   moe_intermediate_size = 2560
//   num_experts           = 64
//   num_experts_per_tok   = 8   (top_k)
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states            : [batch_size, seq_len, hidden_size]               bfloat16
//     gate_weight              : [num_experts, hidden_size]                       bfloat16
//     expert_gate_proj_weights : [num_experts, moe_intermediate_size, hidden_size] bfloat16
//     expert_up_proj_weights   : [num_experts, moe_intermediate_size, hidden_size] bfloat16
//     expert_down_proj_weights : [num_experts, hidden_size, moe_intermediate_size] bfloat16
//   outputs (bfloat16, DPS pre-alloc, written in place):
//     final_hidden_states : [batch_size, seq_len, hidden_size]
//     router_logits       : [total_tokens, num_experts]
//
// Math the reference computes:
//   hidden_states_flat = hidden_states.view(-1, hidden_dim)
//   router_logits      = linear(hidden_states_flat, gate_weight)      (bf16)
//   routing_weights    = softmax(router_logits.float(), dim=1)        (fp32)
//   rw_topk, sel       = topk(routing_weights, top_k=8, dim=-1)
//   rw_topk            = rw_topk / rw_topk.sum(-1, keepdim)           (fp32)
//   rw_topk            = rw_topk.to(bf16)
//   expert_mask        = one_hot(sel, num_experts).permute(2, 1, 0)
//                        -> [num_experts, top_k, total_tokens]
//   per expert e in range(num_experts):
//     idx, top_x = where(expert_mask[e])          (slot index, token index)
//     skip e when no token routes to it (numel guard)
//     current_state = hidden_states_flat[top_x]                       (bf16)
//     gate_out  = linear(current_state, gate_proj_w)                  (bf16)
//     up_out    = linear(current_state, up_proj_w)                    (bf16)
//     silu_gate = gate_out / (1 + exp(-gate_out.float())).to(bf16)
//     intermediate = silu_gate * up_out                              (bf16)
//     cur_hidden = linear(intermediate, down_proj_w)                  (bf16)
//     cur_hidden = cur_hidden * rw_topk[top_x, idx, None]            (bf16)
//     final_flat.index_add_(0, top_x, cur_hidden.to(bf16))
//   final_hidden_states = final_flat.reshape(batch_size, seq_len, hidden_dim)
//   return final_hidden_states, router_logits.to(bf16)
//
// Precision mirrors the reference exactly. Only the routing path runs in fp32:
// the gate linear's bf16 logits are upcast to fp32 for the softmax, then top-k
// selection / normalization stay in fp32 and the normalized top-k weights are
// cast back to bf16. Every expert-MLP matmul (gate / up / down projections)
// runs in bf16 just as the reference's F.linear calls do -- no fp32 upcast on
// the expert path. The SiLU follows the reference's exact mixed form: the
// sigmoid denominator (1 + exp(-gate_out)) is evaluated in fp32 and cast back
// to bf16, and the bf16 gate_out is divided by that bf16 denominator. The
// per-(token, slot) routing weight multiply and the index_add_ contribution
// are bf16. The first output is the flat accumulator reshaped to the input's
// 3D layout; the second output is the bf16-cast router logits (the pre-softmax
// gate output). The host loop runs over the fixed expert count and issues work
// only on the current PyTorch stream.

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& gate_weight,
    const torch::Tensor& expert_gate_proj_weights,
    const torch::Tensor& expert_up_proj_weights,
    const torch::Tensor& expert_down_proj_weights,
    torch::Tensor final_hidden_states,
    torch::Tensor router_logits
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(gate_weight, "gate_weight");
    check_bf16_cuda(expert_gate_proj_weights, "expert_gate_proj_weights");
    check_bf16_cuda(expert_up_proj_weights, "expert_up_proj_weights");
    check_bf16_cuda(expert_down_proj_weights, "expert_down_proj_weights");
    check_bf16_cuda(final_hidden_states, "final_hidden_states");
    check_bf16_cuda(router_logits, "router_logits");

    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be 3D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(gate_weight.dim() == 2, "gate_weight must be 2D, got ", gate_weight.dim(), "D");
    TORCH_CHECK(expert_gate_proj_weights.dim() == 3, "expert_gate_proj_weights must be 3D, got ", expert_gate_proj_weights.dim(), "D");
    TORCH_CHECK(expert_up_proj_weights.dim() == 3, "expert_up_proj_weights must be 3D, got ", expert_up_proj_weights.dim(), "D");
    TORCH_CHECK(expert_down_proj_weights.dim() == 3, "expert_down_proj_weights must be 3D, got ", expert_down_proj_weights.dim(), "D");
    TORCH_CHECK(final_hidden_states.dim() == 3, "final_hidden_states must be 3D, got ", final_hidden_states.dim(), "D");
    TORCH_CHECK(router_logits.dim() == 2, "router_logits must be 2D, got ", router_logits.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t hidden_dim = hidden_states.size(2);
    const int64_t num_experts = gate_weight.size(0);
    constexpr int64_t kTopK = 8;

    // Flatten tokens for routing: [total_tokens, hidden_dim].
    auto hidden_states_flat = hidden_states.view({-1, hidden_dim});

    // Gate routing logits (bf16, as F.linear on bf16 inputs).
    auto logits = at::linear(hidden_states_flat, gate_weight);  // [T, num_experts]

    // Softmax routing weights in fp32, then top-k selection and normalization.
    auto routing_weights = at::softmax(logits.to(torch::kFloat32), /*dim=*/1);
    auto topk = at::topk(routing_weights, /*k=*/kTopK, /*dim=*/-1,
                         /*largest=*/true, /*sorted=*/true);
    auto routing_weights_topk = std::get<0>(topk);              // [T, top_k] fp32
    auto selected_experts = std::get<1>(topk);                  // [T, top_k] int64
    routing_weights_topk = routing_weights_topk /
        routing_weights_topk.sum(/*dim=*/-1, /*keepdim=*/true);
    routing_weights_topk = routing_weights_topk.to(torch::kBFloat16);

    // Flat accumulator [total_tokens, hidden_dim], zero-initialized.
    auto accum = at::zeros({batch_size * seq_len, hidden_dim},
                           hidden_states.options());

    // expert_mask: [num_experts, top_k, total_tokens].
    auto expert_mask =
        at::one_hot(selected_experts, num_experts).permute({2, 1, 0});

    for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        // where on the [top_k, total_tokens] slice returns {slot indices,
        // token indices} in row-major order -- matching the reference's
        // (idx, top_x) unpacking.
        auto nz = at::where(expert_mask[expert_idx]);
        auto idx = nz[0];     // slot indices
        auto top_x = nz[1];   // token indices

        if (top_x.numel() == 0) {
            continue;
        }

        // Per-expert projection weights.
        auto gate_proj_w = expert_gate_proj_weights[expert_idx];  // [intermediate, hidden]
        auto up_proj_w = expert_up_proj_weights[expert_idx];      // [intermediate, hidden]
        auto down_proj_w = expert_down_proj_weights[expert_idx];  // [hidden, intermediate]

        // Gather this expert's tokens (bf16, no upcast on the expert path).
        auto current_state = hidden_states_flat.index_select(/*dim=*/0, top_x);

        // Gate / up projections in bf16.
        auto gate_out = at::linear(current_state, gate_proj_w);  // [num_tokens, intermediate]
        auto up_out = at::linear(current_state, up_proj_w);      // [num_tokens, intermediate]

        // SiLU in the reference's mixed form: fp32 sigmoid denominator cast
        // back to bf16, then the bf16 gate_out divided by it.
        auto silu_gate = gate_out /
            (1.0 + at::exp(-gate_out.to(torch::kFloat32))).to(torch::kBFloat16);
        auto intermediate = silu_gate * up_out;                  // [num_tokens, intermediate]

        // Down projection.
        auto current_hidden_states = at::linear(intermediate, down_proj_w);

        // Weight by the per-(token, slot) bf16 routing weight, then accumulate.
        auto weights = routing_weights_topk.index({top_x, idx}).unsqueeze(-1);
        current_hidden_states = current_hidden_states * weights;
        accum.index_add_(
            /*dim=*/0, top_x, current_hidden_states.to(torch::kBFloat16));
    }

    // First output: flat accumulator reshaped to the input's 3D layout.
    auto final_3d = accum.reshape({batch_size, seq_len, hidden_dim});
    TORCH_CHECK(final_hidden_states.sizes() == final_3d.sizes(),
                "final_hidden_states shape ", final_hidden_states.sizes(),
                " does not match computed ", final_3d.sizes());
    final_hidden_states.copy_(final_3d);

    // Second output: bf16-cast router logits (pre-softmax gate output).
    auto router_logits_out = logits.to(torch::kBFloat16);
    TORCH_CHECK(router_logits.sizes() == router_logits_out.sizes(),
                "router_logits shape ", router_logits.sizes(),
                " does not match computed ", router_logits_out.sizes());
    router_logits.copy_(router_logits_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "MoE expert MLP with internal load-balancing router. Pure ATen "
        "dispatch: bf16 gate linear -> fp32 softmax -> top-8 selection and "
        "fp32 normalization (cast back to bf16); per-expert host loop over the "
        "fixed expert count gathers each expert's tokens (one_hot / permute / "
        "where / index_select), runs gate / up / down projections and the "
        "mixed-precision SiLU SwiGLU gate in bf16, weights by the bf16 "
        "per-token routing weight, and accumulates via index_add_, reproduced "
        "op-for-op from the reference. Two outputs (destination-passing style): "
        "the reshaped final hidden states and the bf16 router logits.");
}
