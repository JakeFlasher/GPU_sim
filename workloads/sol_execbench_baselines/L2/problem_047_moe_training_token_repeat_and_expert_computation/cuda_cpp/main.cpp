#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// MoE training forward pass with token repetition and masked expert
// computation. Each token is repeated num_experts_per_tok times, dispatched to
// its selected experts through a masked SwiGLU FFN, reshaped, and combined with
// the routing weights summed across experts. Reproduces the reference run()
// op-for-op via pure ATen / cuBLAS dispatch -- no custom device kernels.
//
// Dimensions from the selected problem definition:
//   hidden_size           = 8192
//   moe_intermediate_size = 2048
//   num_experts           = definition-selected (read from projection tensors)
//   num_experts_per_tok   = 8
//
// Per definition.json (insertion order):
//   inputs (all float16 except topk_idx):
//     hidden_states     : [batch_seq_len, hidden_size]                       float16
//     topk_idx          : [batch_seq_len, num_experts_per_tok]               int64
//     topk_weight       : [batch_seq_len, num_experts_per_tok]               float16
//     expert_gate_projs : [num_experts, moe_intermediate_size, hidden_size]  float16
//     expert_up_projs   : [num_experts, moe_intermediate_size, hidden_size]  float16
//     expert_down_projs : [num_experts, hidden_size, moe_intermediate_size]  float16
//   output (float16, DPS pre-alloc, written in place):
//     output : [batch_seq_len, hidden_size]
//
// Math the reference computes:
//   hidden_states_repeated = hidden_states.repeat_interleave(num_experts_per_tok, dim=0)
//                            -> [batch_seq_len * num_experts_per_tok, hidden]
//   flat_topk_idx          = topk_idx.view(-1)
//   y = zeros_like(hidden_states_repeated)
//   per expert e in range(num_experts):
//     expert_mask = (flat_topk_idx == e)
//     skip e when no row matches (any() guard)
//     expert_input  = hidden_states_repeated[expert_mask]                    [n_e, hidden]
//     gate_output   = linear(expert_input, gate_w[e])                        [n_e, inter]
//     up_output     = linear(expert_input, up_w[e])                          [n_e, inter]
//     intermediate  = silu(gate_output) * up_output                          [n_e, inter]
//     expert_output = linear(intermediate, down_w[e])                        [n_e, hidden]
//     y[expert_mask] = expert_output
//   y = y.view(batch_seq_len, num_experts_per_tok, hidden)
//   output = (y * topk_weight.unsqueeze(-1)).sum(dim=1)
//
// Precision profile (matched exactly): every tensor is float16 and the
// reference stays in float16 throughout -- the gate / up / down projections,
// the SiLU SwiGLU gate, the routing-weight multiply, and the cross-expert sum
// all run in fp16 just as the reference's F.linear / F.silu / elementwise ops
// do. There are no fp32 upcast points. The per-expert weight slice is sliced
// inside the loop so the resident footprint stays at one expert's weights
// rather than the whole [num_experts, ...] stack. The boolean-mask gather /
// scatter is expressed as nonzero -> index_select / index_copy_ over the masked
// rows, reproducing the reference's hidden_states_repeated[mask] read and
// y[mask] = ... write exactly. The host loop runs over the fixed expert count
// read from expert_gate_projs.size(0) (iterating every expert and skipping the
// empty ones yields the identical set of writes) and issues work only on the
// current PyTorch stream.

static void check_fp16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat16, name, " must be float16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weight,
    const torch::Tensor& expert_gate_projs,
    const torch::Tensor& expert_up_projs,
    const torch::Tensor& expert_down_projs,
    torch::Tensor output
) {
    check_fp16_cuda(hidden_states, "hidden_states");
    TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
    TORCH_CHECK(topk_idx.dtype() == torch::kInt64,
                "topk_idx must be int64, got ", topk_idx.dtype());
    check_fp16_cuda(topk_weight, "topk_weight");
    check_fp16_cuda(expert_gate_projs, "expert_gate_projs");
    check_fp16_cuda(expert_up_projs, "expert_up_projs");
    check_fp16_cuda(expert_down_projs, "expert_down_projs");
    check_fp16_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must be 2D, got ", topk_idx.dim(), "D");
    TORCH_CHECK(topk_weight.dim() == 2, "topk_weight must be 2D, got ", topk_weight.dim(), "D");
    TORCH_CHECK(expert_gate_projs.dim() == 3, "expert_gate_projs must be 3D, got ", expert_gate_projs.dim(), "D");
    TORCH_CHECK(expert_up_projs.dim() == 3, "expert_up_projs must be 3D, got ", expert_up_projs.dim(), "D");
    TORCH_CHECK(expert_down_projs.dim() == 3, "expert_down_projs must be 3D, got ", expert_down_projs.dim(), "D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D, got ", output.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t batch_seq_len = hidden_states.size(0);
    const int64_t hidden_size = hidden_states.size(1);
    const int64_t num_experts = expert_gate_projs.size(0);
    const int64_t num_experts_per_tok = topk_idx.size(1);

    // Repeat each token once per selected expert, then flatten the routed
    // expert ids to a single dispatch axis.
    auto hidden_states_repeated =
        hidden_states.repeat_interleave(num_experts_per_tok, /*dim=*/0);  // [B*k, hidden]
    auto flat_topk_idx = topk_idx.view({-1});                             // [B*k]

    // Per-expert outputs scatter into a zero-initialized dispatch buffer.
    auto y = at::zeros_like(hidden_states_repeated);

    for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        // Rows of the dispatch buffer routed to this expert.
        auto expert_mask = (flat_topk_idx == expert_idx);
        auto rows = at::nonzero(expert_mask).squeeze(-1);

        if (rows.numel() == 0) {
            continue;
        }

        // Per-expert projection weights.
        auto gate_w = expert_gate_projs[expert_idx];  // [inter, hidden]
        auto up_w = expert_up_projs[expert_idx];      // [inter, hidden]
        auto down_w = expert_down_projs[expert_idx];  // [hidden, inter]

        // Gather this expert's tokens (boolean-mask read of the dispatch buffer).
        auto expert_input = hidden_states_repeated.index_select(/*dim=*/0, rows);  // [n_e, hidden]

        // SwiGLU FFN, native fp16 throughout.
        auto gate_output = at::linear(expert_input, gate_w);  // [n_e, inter]
        auto up_output = at::linear(expert_input, up_w);      // [n_e, inter]
        auto intermediate = at::silu(gate_output) * up_output;  // [n_e, inter]
        auto expert_output = at::linear(intermediate, down_w);  // [n_e, hidden]

        // Write back to the routed rows (boolean-mask assignment).
        y.index_copy_(/*dim=*/0, rows, expert_output);
    }

    // Reshape to [batch_seq_len, num_experts_per_tok, hidden], apply the
    // routing weights, and sum across the per-token expert axis.
    auto y_reshaped = y.view({batch_seq_len, num_experts_per_tok, hidden_size});
    auto weighted = y_reshaped * topk_weight.unsqueeze(-1);
    auto result = weighted.sum(/*dim=*/std::vector<int64_t>{1});  // [B, hidden]

    // Destination-passing-style write.
    TORCH_CHECK(output.sizes() == result.sizes(),
                "output shape ", output.sizes(),
                " does not match computed ", result.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "MoE training forward with token repetition and masked expert dispatch "
        "(expert count derived from the projection tensors). Pure ATen dispatch: "
        "repeat_interleave each token once "
        "per selected expert, then a per-expert host loop over the fixed expert "
        "count gathers the routed rows (mask -> nonzero / index_select), runs "
        "the gate / up / down SwiGLU FFN in native float16, and scatters the "
        "expert output back (index_copy_); the dispatch buffer is reshaped, "
        "weighted by the fp16 routing weights, and summed across the per-token "
        "expert axis, reproduced op-for-op from the reference. Single output "
        "(destination-passing style), native float16 throughout.");
}
