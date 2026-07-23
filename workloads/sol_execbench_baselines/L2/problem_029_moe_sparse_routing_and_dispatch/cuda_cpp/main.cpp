#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// MoE sparse routing and dispatch: a shared expert over all tokens, a softmax
// router (fp32) with correction bias and top-8 selection from 64 experts,
// clamped weight normalization, SwiGLU expert MLPs with weighted aggregation,
// and a sum with the shared branch. Emits both the routed-plus-shared output
// and the raw fp32 router logits. Reproduces the reference run() op-for-op via
// pure ATen / cuBLAS dispatch -- no custom device kernels.
//
// Pinned axes from the problem definition:
//   hidden_size              = 8192
//   num_experts              = 64
//   top_k                    = 8
//   intermediate_size        = 3584
//   shared_intermediate_size = 7168
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states            : [batch_size, seq_len, hidden_size]            bfloat16
//     gate_weight              : [num_experts, hidden_size]                    float32
//     e_score_correction_bias  : [1, num_experts]                             float32
//     expert_gate_proj         : [num_experts, intermediate, hidden_size]      bfloat16
//     expert_up_proj           : [num_experts, intermediate, hidden_size]      bfloat16
//     expert_down_proj         : [num_experts, hidden_size, intermediate]      bfloat16
//     shared_gate_proj         : [shared_intermediate, hidden_size]            bfloat16
//     shared_up_proj           : [shared_intermediate, hidden_size]            bfloat16
//     shared_down_proj         : [hidden_size, shared_intermediate]            bfloat16
//     norm_min                 : scalar                                        float32
//   outputs (DPS pre-alloc, written in place):
//     output        : [batch_size, seq_len, hidden_size]   bfloat16
//     router_logits : [batch_seq_len, num_experts]         float32
//
// The trailing norm_min input carries shape None in the definition, so the
// evaluator marshals it as a plain Python scalar rather than a tensor; it is
// received here by value as a double.
//
// Math the reference computes:
//   hidden_states_flat = hidden_states.view(-1, hidden_dim)
//   -- shared expert over all tokens, bf16:
//   shared_gate_out = hidden_states_flat @ shared_gate_proj.t()                 (bf16)
//   shared_inter    = silu(shared_gate_out) * (hidden_states_flat @ shared_up_proj.t())
//   shared_output   = shared_inter @ shared_down_proj.t()                       (bf16)
//   -- router in fp32:
//   router_logits   = hidden_states_flat.float() @ gate_weight.t()             (fp32)
//   routing_weights = softmax(router_logits, dim=1, dtype=fp32)
//   routing_weights = routing_weights + e_score_correction_bias               (fp32)
//   _, selected_experts = topk(routing_weights, top_k=8, dim=-1)
//   sel             = gather(routing_weights, -1, selected_experts)            (fp32)
//   normalized      = sel / clamp(sel.sum(-1, keepdim), min=norm_min)          (fp32)
//   normalized      = normalized.to(bf16)
//   final = zeros([batch_seq_len, hidden_dim], bf16)
//   expert_mask = one_hot(selected_experts, 64).permute(2, 1, 0)
//                 -> [num_experts, top_k, batch_seq_len]
//   expert_hit = where expert_mask.sum((-1, -2)) > 0; loop over those experts.
//   per hit expert e:
//     idx, top_x = where(expert_mask[e].squeeze(0))   (slot index, token index)
//     current_state = hidden_states_flat[top_x]                                (bf16)
//     gate_out  = current_state @ expert_gate_proj[e].t()                      (bf16)
//     up_out    = current_state @ expert_up_proj[e].t()                        (bf16)
//     intermediate = silu(gate_out) * up_out                                   (bf16)
//     expert_output = intermediate @ expert_down_proj[e].t()                   (bf16)
//     weighted  = expert_output * normalized[top_x, idx, None]                 (bf16)
//     final.index_add_(0, top_x, weighted)
//   final = final + shared_output                                              (bf16)
//   output = final.view(batch_size, seq_len, hidden_dim)
//   return output, router_logits
//
// Precision mirrors the reference exactly. Only the router runs in fp32: the
// gate matmul takes fp32-upcast hidden states against the fp32 gate weight to
// yield the fp32 router logits (the second output, returned unmodified), then
// the softmax / bias add / top-k / gather / clamped normalization all stay
// fp32 before the normalized weights are cast back to bf16. The shared expert
// and every routed expert MLP -- gate / up / down projections and the SiLU
// SwiGLU gate -- run in bf16, matching the reference's matmul / F.silu calls
// on the bf16 weights. The per-(token, slot) weight multiply and the
// index_add_ contribution are bf16, and the final routed-plus-shared sum is
// bf16. Iterating every expert with an any() guard issues the identical set of
// accumulations as the reference's iteration over the precomputed hit set,
// while avoiding the data-dependent per-iteration host sync the reference
// .item() loop incurs. The host loop issues work only on the current PyTorch
// stream.

static void check_dtype_cuda(const torch::Tensor& t,
                             torch::ScalarType dt,
                             const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == dt, name, " has wrong dtype, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& gate_weight,
    const torch::Tensor& e_score_correction_bias,
    const torch::Tensor& expert_gate_proj,
    const torch::Tensor& expert_up_proj,
    const torch::Tensor& expert_down_proj,
    const torch::Tensor& shared_gate_proj,
    const torch::Tensor& shared_up_proj,
    const torch::Tensor& shared_down_proj,
    double norm_min,
    torch::Tensor output,
    torch::Tensor router_logits
) {
    check_dtype_cuda(hidden_states, torch::kBFloat16, "hidden_states");
    check_dtype_cuda(gate_weight, torch::kFloat32, "gate_weight");
    check_dtype_cuda(e_score_correction_bias, torch::kFloat32, "e_score_correction_bias");
    check_dtype_cuda(expert_gate_proj, torch::kBFloat16, "expert_gate_proj");
    check_dtype_cuda(expert_up_proj, torch::kBFloat16, "expert_up_proj");
    check_dtype_cuda(expert_down_proj, torch::kBFloat16, "expert_down_proj");
    check_dtype_cuda(shared_gate_proj, torch::kBFloat16, "shared_gate_proj");
    check_dtype_cuda(shared_up_proj, torch::kBFloat16, "shared_up_proj");
    check_dtype_cuda(shared_down_proj, torch::kBFloat16, "shared_down_proj");
    check_dtype_cuda(output, torch::kBFloat16, "output");
    check_dtype_cuda(router_logits, torch::kFloat32, "router_logits");

    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be 3D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(gate_weight.dim() == 2, "gate_weight must be 2D, got ", gate_weight.dim(), "D");
    TORCH_CHECK(e_score_correction_bias.dim() == 2, "e_score_correction_bias must be 2D, got ", e_score_correction_bias.dim(), "D");
    TORCH_CHECK(expert_gate_proj.dim() == 3, "expert_gate_proj must be 3D, got ", expert_gate_proj.dim(), "D");
    TORCH_CHECK(expert_up_proj.dim() == 3, "expert_up_proj must be 3D, got ", expert_up_proj.dim(), "D");
    TORCH_CHECK(expert_down_proj.dim() == 3, "expert_down_proj must be 3D, got ", expert_down_proj.dim(), "D");
    TORCH_CHECK(shared_gate_proj.dim() == 2, "shared_gate_proj must be 2D, got ", shared_gate_proj.dim(), "D");
    TORCH_CHECK(shared_up_proj.dim() == 2, "shared_up_proj must be 2D, got ", shared_up_proj.dim(), "D");
    TORCH_CHECK(shared_down_proj.dim() == 2, "shared_down_proj must be 2D, got ", shared_down_proj.dim(), "D");
    TORCH_CHECK(output.dim() == 3, "output must be 3D, got ", output.dim(), "D");
    TORCH_CHECK(router_logits.dim() == 2, "router_logits must be 2D, got ", router_logits.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t hidden_dim = hidden_states.size(2);
    const int64_t num_experts = gate_weight.size(0);
    constexpr int64_t kTopK = 8;

    // Flatten tokens: [batch_seq_len, hidden_dim].
    auto hidden_states_flat = hidden_states.view({-1, hidden_dim});

    // Shared expert over all tokens, bf16 (computed first, as in the reference).
    auto shared_gate_out = at::matmul(hidden_states_flat, shared_gate_proj.t());
    auto shared_silu = at::silu(shared_gate_out);
    auto shared_up_out = at::matmul(hidden_states_flat, shared_up_proj.t());
    auto shared_intermediate = shared_silu * shared_up_out;
    auto shared_output = at::matmul(shared_intermediate, shared_down_proj.t());

    // Router logits in fp32: fp32-upcast tokens against the fp32 gate weight.
    auto logits =
        at::matmul(hidden_states_flat.to(torch::kFloat32), gate_weight.t());

    // Softmax routing probabilities, correction bias, and top-k selection, fp32.
    auto routing_weights = at::softmax(logits, /*dim=*/1);
    routing_weights = routing_weights + e_score_correction_bias;
    auto topk = at::topk(routing_weights, /*k=*/kTopK, /*dim=*/-1,
                         /*largest=*/true, /*sorted=*/true);
    auto selected_experts = std::get<1>(topk);    // [T, top_k] int64

    // Gather the selected experts' weights and normalize with min-clamping.
    auto routing_weights_selected =
        at::gather(routing_weights, /*dim=*/-1, selected_experts);
    auto denominator = at::clamp_min(
        routing_weights_selected.sum(/*dim=*/-1, /*keepdim=*/true), norm_min);
    auto routing_weights_normalized = routing_weights_selected / denominator;
    routing_weights_normalized = routing_weights_normalized.to(torch::kBFloat16);

    // Flat accumulator [batch_seq_len, hidden_dim], bf16, zero-initialized.
    auto final_hidden_states =
        at::zeros({batch_size * seq_len, hidden_dim}, hidden_states.options());

    // expert_mask: [num_experts, top_k, batch_seq_len].
    auto expert_mask =
        at::one_hot(selected_experts, num_experts).permute({2, 1, 0});

    for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        auto mask = expert_mask[expert_idx];   // [top_k, batch_seq_len]

        // Iterate every expert but issue work only for hit experts -- the same
        // set the reference's expert_hit nonzero loop processes.
        if (!mask.any().item<bool>()) {
            continue;
        }

        // where on the [top_k, batch_seq_len] slice returns {slot indices,
        // token indices} in row-major order -- matching the reference's
        // (idx, top_x) unpacking.
        auto nz = at::where(mask);
        auto idx = nz[0];     // slot indices
        auto top_x = nz[1];   // token indices

        if (top_x.numel() == 0) {
            continue;
        }

        // Per-expert projection weights.
        auto gate_w = expert_gate_proj[expert_idx];   // [intermediate, hidden]
        auto up_w = expert_up_proj[expert_idx];        // [intermediate, hidden]
        auto down_w = expert_down_proj[expert_idx];    // [hidden, intermediate]

        // Gather this expert's tokens (bf16, no upcast on the expert path).
        auto current_state = hidden_states_flat.index_select(/*dim=*/0, top_x);

        // SwiGLU expert MLP in bf16, matching the reference's matmul / F.silu.
        auto gate_out = at::matmul(current_state, gate_w.t());
        auto silu_out = at::silu(gate_out);
        auto up_out = at::matmul(current_state, up_w.t());
        auto intermediate = silu_out * up_out;
        auto expert_output = at::matmul(intermediate, down_w.t());

        // Weight by the bf16 per-(token, slot) routing weight, then accumulate.
        auto weights = routing_weights_normalized.index({top_x, idx}).unsqueeze(-1);
        auto weighted_output = expert_output * weights;
        final_hidden_states.index_add_(/*dim=*/0, top_x, weighted_output);
    }

    // Add the shared-expert branch (bf16) and reshape to the input layout.
    final_hidden_states = final_hidden_states + shared_output;
    auto output_3d =
        final_hidden_states.view({batch_size, seq_len, hidden_dim});

    // First output: the routed-plus-shared hidden states.
    TORCH_CHECK(output.sizes() == output_3d.sizes(),
                "output shape ", output.sizes(),
                " does not match computed ", output_3d.sizes());
    output.copy_(output_3d);

    // Second output: the raw fp32 router logits (returned unmodified).
    TORCH_CHECK(router_logits.sizes() == logits.sizes(),
                "router_logits shape ", router_logits.sizes(),
                " does not match computed ", logits.sizes());
    router_logits.copy_(logits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "MoE sparse routing and dispatch with shared expert. Pure ATen "
        "dispatch: bf16 shared expert over all tokens, fp32 softmax router "
        "with correction bias and top-8 selection (fp32 gather and "
        "min-clamped normalization, cast back to bf16); per-expert host loop "
        "over the fixed expert count with an any() guard gathers each hit "
        "expert's tokens (one_hot / permute / where / index_select), runs the "
        "SwiGLU expert MLP in bf16, weights by the bf16 per-token routing "
        "weight and accumulates via index_add_, then adds the shared branch, "
        "reproduced op-for-op from the reference. Two outputs "
        "(destination-passing style): the reshaped final hidden states and the "
        "fp32 router logits.");
}
