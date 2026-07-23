#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Optimized MoE expert inference dispatch: batch tokens by expert assignment,
// run each expert's SwiGLU MLP over its contiguous slice of sorted tokens, then
// scatter the per-token-expert outputs back and combine them with their routing
// weights. Reproduces the reference run() op-for-op via pure ATen / cuBLAS
// dispatch -- no custom device kernels. This is a 256-expert system with 8
// experts selected per token.
//
// Pinned axes from the problem definition:
//   hidden_size         = 4096
//   intermediate_size   = 1024
//   num_experts         = 256
//   num_experts_per_tok = 8
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states : [num_tokens, hidden_size]                   bfloat16
//     topk_idx      : [num_tokens, num_experts_per_tok]           int64
//     topk_weight   : [num_tokens, num_experts_per_tok]           bfloat16
//     gate_weights  : [num_experts, intermediate_size, hidden]    bfloat16
//     up_weights    : [num_experts, intermediate_size, hidden]    bfloat16
//     down_weights  : [num_experts, hidden, intermediate_size]    bfloat16
//   output (bfloat16, DPS pre-alloc, written in place):
//     output : [num_tokens, hidden_size]
//
// The reference upcasts every operand to float32 for the expert math and only
// narrows back to bfloat16 on the final result, so the entire expert path here
// runs in float32 and the single trailing cast to bfloat16 mirrors the
// reference's one narrowing point. The dispatch is reproduced exactly: count
// tokens per expert with a scatter-add over topk_idx, argsort the flattened
// expert assignments to get a stable expert-major ordering, gather the source
// tokens into sorted order, then a host loop over the fixed expert count slices
// the contiguous [start, end) range belonging to each expert and runs its
// gate / up / SwiGLU / down projections. The per-expert token counts are read
// to host with .item() (the reference's tokens_per_expert.cpu()) to drive the
// slice boundaries. Expert outputs are scattered back to their token-expert
// rows via the same permutation, reshaped to [num_tokens, top_k, hidden],
// weighted by the per-(token, slot) routing weight, and summed across the slot
// axis. All device work is issued on the current PyTorch stream.

static void check_dtype_cuda(const torch::Tensor& t,
                             torch::ScalarType dt,
                             const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == dt, name, " has wrong dtype, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weight,
    const torch::Tensor& gate_weights,
    const torch::Tensor& up_weights,
    const torch::Tensor& down_weights,
    torch::Tensor output
) {
    check_dtype_cuda(hidden_states, torch::kBFloat16, "hidden_states");
    check_dtype_cuda(topk_idx, torch::kInt64, "topk_idx");
    check_dtype_cuda(topk_weight, torch::kBFloat16, "topk_weight");
    check_dtype_cuda(gate_weights, torch::kBFloat16, "gate_weights");
    check_dtype_cuda(up_weights, torch::kBFloat16, "up_weights");
    check_dtype_cuda(down_weights, torch::kBFloat16, "down_weights");
    check_dtype_cuda(output, torch::kBFloat16, "output");

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D, got ", hidden_states.dim(), "D");
    TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must be 2D, got ", topk_idx.dim(), "D");
    TORCH_CHECK(topk_weight.dim() == 2, "topk_weight must be 2D, got ", topk_weight.dim(), "D");
    TORCH_CHECK(gate_weights.dim() == 3, "gate_weights must be 3D, got ", gate_weights.dim(), "D");
    TORCH_CHECK(up_weights.dim() == 3, "up_weights must be 3D, got ", up_weights.dim(), "D");
    TORCH_CHECK(down_weights.dim() == 3, "down_weights must be 3D, got ", down_weights.dim(), "D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D, got ", output.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t num_tokens = hidden_states.size(0);
    const int64_t hidden_size = hidden_states.size(1);
    const int64_t num_experts = gate_weights.size(0);
    const int64_t num_experts_per_tok = topk_idx.size(1);

    // Upcast the activations to float32 for the expert math (the reference's
    // numerical-stability cast). The per-expert weight slices are upcast
    // inside the dispatch loop below: casting a single expert's slice is
    // bit-identical to slicing a full-tensor cast, but keeps only one expert's
    // float32 weights resident instead of the entire [num_experts, ...] set
    // (which, for a large expert count, would not fit alongside the bf16
    // inputs on a memory-bounded device).
    auto hidden_states_f32 = hidden_states.to(torch::kFloat32);
    auto topk_weight_f32 = topk_weight.to(torch::kFloat32);

    // Count tokens per expert: scatter a 1 into each selected expert column,
    // then sum down the token axis. cnts holds at most one mark per
    // (token, expert) pair, matching the reference's scatter_ semantics.
    auto cnts = at::zeros({num_tokens, num_experts},
                          topk_idx.options().dtype(torch::kInt32));
    cnts.scatter_(/*dim=*/1, topk_idx,
                  at::ones({num_tokens, num_experts},
                           topk_idx.options().dtype(torch::kInt32)));
    auto tokens_per_expert = cnts.sum(/*dim=*/0);   // [num_experts] int32

    // Sort the flattened token-expert assignments by expert ID; idxs is a
    // permutation of every token-expert pair into expert-major order.
    auto flat_topk_idx = topk_idx.view({-1});
    auto idxs = flat_topk_idx.argsort();
    auto token_indices = idxs.floor_divide(num_experts_per_tok);
    auto sorted_tokens = hidden_states_f32.index_select(/*dim=*/0, token_indices);

    // Per-expert counts to host to drive the contiguous slice boundaries
    // (the reference's tokens_per_expert.cpu()).
    auto tokens_per_expert_cpu = tokens_per_expert.to(torch::kCPU);

    std::vector<torch::Tensor> outputs;
    outputs.reserve(num_experts);
    int64_t start_idx = 0;

    for (int64_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        const int64_t num_tokens_for_expert =
            tokens_per_expert_cpu[expert_id].item<int64_t>();

        if (num_tokens_for_expert == 0) {
            continue;
        }

        const int64_t end_idx = start_idx + num_tokens_for_expert;

        // This expert's contiguous slice of sorted tokens.
        auto expert_input = sorted_tokens.slice(/*dim=*/0, start_idx, end_idx);

        // SwiGLU MLP: gate / up projections, silu(gate) * up, down projection.
        // Each expert's weight slice is upcast to float32 here (bit-identical
        // to slicing a full-tensor cast) so only one expert's float32 weights
        // are resident at a time.
        auto gate_out = at::matmul(expert_input, gate_weights[expert_id].to(torch::kFloat32).t());
        auto up_out = at::matmul(expert_input, up_weights[expert_id].to(torch::kFloat32).t());
        auto activated = (gate_out * at::sigmoid(gate_out)) * up_out;
        auto expert_output = at::matmul(activated, down_weights[expert_id].to(torch::kFloat32).t());

        outputs.push_back(expert_output);
        start_idx = end_idx;
    }

    auto result = at::zeros({num_tokens, hidden_size}, output.options());
    if (!outputs.empty()) {
        // Concatenate the per-expert outputs in expert-major order, then
        // scatter each row back to its token-expert position via the same
        // permutation (the reference's new_x[idxs] = all_outputs).
        auto all_outputs = at::cat(outputs, /*dim=*/0);
        auto new_x = at::zeros_like(all_outputs);
        new_x.index_copy_(/*dim=*/0, idxs, all_outputs);

        // Reshape to [num_tokens, top_k, hidden], weight by the per-(token,
        // slot) routing weight, and sum across the slot axis.
        new_x = new_x.view({num_tokens, num_experts_per_tok, hidden_size});
        auto weighted_output = new_x * topk_weight_f32.unsqueeze(-1);
        auto final_output = weighted_output.sum(/*dim=*/1);

        result = final_output.to(torch::kBFloat16);
    }

    TORCH_CHECK(output.sizes() == result.sizes(),
                "output shape ", output.sizes(),
                " does not match computed ", result.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "MoE expert inference with batched token dispatch (256 experts, 8 per "
        "token). Pure ATen dispatch: count tokens per expert via scatter-add, "
        "argsort the flattened expert assignments into expert-major order, "
        "gather the source tokens, then a host loop over the fixed expert count "
        "slices each expert's contiguous token range and runs gate / up / "
        "SwiGLU / down projections in float32. Per-expert outputs are scattered "
        "back via the same permutation, reshaped, weighted by the per-token "
        "routing weight, and summed across the selected-expert axis, reproduced "
        "op-for-op from the reference. Destination-passing style (single "
        "bfloat16 output written in place).");
}
