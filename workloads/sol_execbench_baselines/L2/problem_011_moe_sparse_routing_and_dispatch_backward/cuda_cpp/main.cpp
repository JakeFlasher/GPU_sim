#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

static void check_bf16_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kBFloat16,
                name, " must be bfloat16, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static void check_fp32_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32,
                name, " must be float32, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static void check_int64_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kInt64,
                name, " must be int64, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the MoE sparse routing
// backward. Per definition.json:
//   hidden_states           : [N, hidden_size=2048]      bf16
//   gate_weight             : [num_experts=128,
//                              hidden_size=2048]         bf16
//   router_logits           : [N, num_experts=128]       bf16
//   routing_probs           : [N, num_experts]           fp32
//   selected_experts        : [N, top_k=8]               int64
//   routing_weights_sum     : [N, 1]                     fp32
//   grad_routing_weights    : [N, top_k]                 bf16
//   grad_expert_mask        : [num_experts, top_k, N]    bf16
//   grad_router_logits      : [N, num_experts]           bf16
//   grad_hidden_states      : [N, hidden_size]           bf16 DPS
//   grad_gate_weight        : [num_experts, hidden_size] bf16 DPS
//
// Algorithm (mirrors the reference exactly):
//   Normalization backward (quotient rule on
//   grad_routing_weights, in fp32):
//     routing_weights_unnorm = routing_probs.gather(1, selected_experts)
//     grad_sum = (grad_routing_weights_f32 * routing_weights_unnorm
//                 / routing_weights_sum).sum(-1, keepdim=true)
//     grad_routing_weights_unnorm =
//       grad_routing_weights_f32 / routing_weights_sum
//       - grad_sum / routing_weights_sum
//   Top-k straight-through scatter back into routing_probs space:
//     grad_routing_probs = zeros_like(routing_probs)  (fp32)
//     grad_routing_probs.scatter_(1, selected_experts,
//                                 grad_routing_weights_unnorm)
//   Auxiliary grad_expert_mask path:
//     grad_expert_mask_permuted = grad_expert_mask
//                                  .permute(2, 1, 0).float()
//     grad_from_mask = gather(grad_expert_mask_permuted, 2,
//                              selected_experts.unsqueeze(2))
//                        .squeeze(2)
//     grad_routing_probs.scatter_add_(1, selected_experts,
//                                     grad_from_mask)
//   Direct grad_router_logits path:
//     grad_routing_probs += grad_router_logits.float()
//   Softmax backward:
//     dot_product = (grad_routing_probs * routing_probs)
//                     .sum(dim=1, keepdim=true)
//     grad_router_logits_computed = routing_probs *
//                                   (grad_routing_probs - dot_product)
//   Cast back to bf16 and dispatch 2D bf16 sgemm pair:
//     grad_router_logits_bf16 = grad_router_logits_computed.to(bf16)
//     grad_hidden_states = grad_router_logits_bf16 @ gate_weight
//                          (cublasGemmEx 2D bf16 OP_N OP_N -- verified)
//     grad_gate_weight = grad_router_logits_bf16.t() @ hidden_states
//                        (cublasGemmEx 2D bf16 OP_T OP_N -- same family)
void run(
    torch::Tensor hidden_states,
    torch::Tensor gate_weight,
    torch::Tensor router_logits,
    torch::Tensor routing_probs,
    torch::Tensor selected_experts,
    torch::Tensor routing_weights_sum,
    torch::Tensor grad_routing_weights,
    torch::Tensor grad_expert_mask,
    torch::Tensor grad_router_logits,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_gate_weight
) {
    check_bf16_contiguous_cuda(hidden_states, "hidden_states");
    check_bf16_contiguous_cuda(gate_weight, "gate_weight");
    check_bf16_contiguous_cuda(router_logits, "router_logits");
    check_fp32_contiguous_cuda(routing_probs, "routing_probs");
    check_int64_contiguous_cuda(selected_experts, "selected_experts");
    check_fp32_contiguous_cuda(routing_weights_sum,
                               "routing_weights_sum");
    check_bf16_contiguous_cuda(grad_routing_weights,
                               "grad_routing_weights");
    check_bf16_contiguous_cuda(grad_expert_mask, "grad_expert_mask");
    check_bf16_contiguous_cuda(grad_router_logits, "grad_router_logits");
    check_bf16_contiguous_cuda(grad_hidden_states, "grad_hidden_states");
    check_bf16_contiguous_cuda(grad_gate_weight, "grad_gate_weight");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kHiddenSize = 2048;
    constexpr int64_t kNumExperts = 128;
    constexpr int64_t kTopK = 8;

    TORCH_CHECK(hidden_states.dim() == 2 &&
                hidden_states.size(1) == kHiddenSize,
                "hidden_states must be [N, hidden_size=2048]");
    const int64_t N = hidden_states.size(0);

    TORCH_CHECK(gate_weight.dim() == 2 &&
                gate_weight.size(0) == kNumExperts &&
                gate_weight.size(1) == kHiddenSize,
                "gate_weight must be [num_experts=128, hidden_size]");
    TORCH_CHECK(router_logits.dim() == 2 &&
                router_logits.size(0) == N &&
                router_logits.size(1) == kNumExperts,
                "router_logits must be [N, num_experts]");
    TORCH_CHECK(routing_probs.dim() == 2 &&
                routing_probs.size(0) == N &&
                routing_probs.size(1) == kNumExperts,
                "routing_probs must be [N, num_experts]");
    TORCH_CHECK(selected_experts.dim() == 2 &&
                selected_experts.size(0) == N &&
                selected_experts.size(1) == kTopK,
                "selected_experts must be [N, top_k=8]");
    TORCH_CHECK(routing_weights_sum.dim() == 2 &&
                routing_weights_sum.size(0) == N &&
                routing_weights_sum.size(1) == 1,
                "routing_weights_sum must be [N, 1]");
    TORCH_CHECK(grad_routing_weights.dim() == 2 &&
                grad_routing_weights.size(0) == N &&
                grad_routing_weights.size(1) == kTopK,
                "grad_routing_weights must be [N, top_k]");
    TORCH_CHECK(grad_expert_mask.dim() == 3 &&
                grad_expert_mask.size(0) == kNumExperts &&
                grad_expert_mask.size(1) == kTopK &&
                grad_expert_mask.size(2) == N,
                "grad_expert_mask must be [num_experts, top_k, N]");
    TORCH_CHECK(grad_router_logits.dim() == 2 &&
                grad_router_logits.size(0) == N &&
                grad_router_logits.size(1) == kNumExperts,
                "grad_router_logits must be [N, num_experts]");

    TORCH_CHECK(grad_hidden_states.dim() == 2 &&
                grad_hidden_states.size(0) == N &&
                grad_hidden_states.size(1) == kHiddenSize,
                "grad_hidden_states must be [N, hidden_size]");
    TORCH_CHECK(grad_gate_weight.dim() == 2 &&
                grad_gate_weight.size(0) == kNumExperts &&
                grad_gate_weight.size(1) == kHiddenSize,
                "grad_gate_weight must be [num_experts, hidden_size]");

    // ----------------------------------------------------------------
    // Normalization backward (quotient rule, fp32 math).
    // ----------------------------------------------------------------
    auto grad_routing_weights_f32 = grad_routing_weights.to(torch::kFloat32);
    auto routing_weights_unnorm = routing_probs.gather(
        /*dim=*/1, selected_experts);

    auto grad_sum = (grad_routing_weights_f32 * routing_weights_unnorm
                     / routing_weights_sum)
                        .sum(/*dim=*/-1, /*keepdim=*/true);
    auto grad_routing_weights_unnorm =
        grad_routing_weights_f32 / routing_weights_sum
        - grad_sum / routing_weights_sum;

    // ----------------------------------------------------------------
    // Top-k straight-through scatter back into routing_probs space.
    // ----------------------------------------------------------------
    auto grad_routing_probs = at::zeros_like(routing_probs);
    grad_routing_probs.scatter_(/*dim=*/1, selected_experts,
                                grad_routing_weights_unnorm);

    // ----------------------------------------------------------------
    // Auxiliary grad_expert_mask path:
    //   permute (num_experts, top_k, N) -> (N, top_k, num_experts),
    //   then gather along dim=2 by selected_experts,
    //   then scatter_add back into routing_probs space.
    // ----------------------------------------------------------------
    auto grad_expert_mask_permuted =
        grad_expert_mask.permute({2, 1, 0}).to(torch::kFloat32);
    auto grad_from_mask = grad_expert_mask_permuted
                              .gather(/*dim=*/2,
                                      selected_experts.unsqueeze(2))
                              .squeeze(2);
    grad_routing_probs.scatter_add_(/*dim=*/1, selected_experts,
                                    grad_from_mask);

    // ----------------------------------------------------------------
    // Direct grad_router_logits path: accumulate into
    // grad_routing_probs.
    // ----------------------------------------------------------------
    grad_routing_probs = grad_routing_probs
                         + grad_router_logits.to(torch::kFloat32);

    // ----------------------------------------------------------------
    // Softmax backward:
    //   d(softmax) = softmax * (grad - sum(grad * softmax))
    // ----------------------------------------------------------------
    auto dot_product = (grad_routing_probs * routing_probs)
                           .sum(/*dim=*/1, /*keepdim=*/true);
    auto grad_router_logits_computed_f32 =
        routing_probs * (grad_routing_probs - dot_product);

    // Cast back to bf16 for the projection backward (matches the
    // reference's `.to(input_dtype)` step).
    auto grad_router_logits_bf16 =
        grad_router_logits_computed_f32.to(torch::kBFloat16);

    // ----------------------------------------------------------------
    // 2D bf16 sgemm dispatch pair on the verified
    // cublasGemmEx_2D_bf16_OP_N_OP_N family.
    // ----------------------------------------------------------------
    grad_hidden_states.copy_(
        at::matmul(grad_router_logits_bf16, gate_weight));
    grad_gate_weight.copy_(
        at::matmul(grad_router_logits_bf16.transpose(0, 1),
                   hidden_states));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "MoE sparse routing backward. Pure ATen / cuBLAS dispatch "
        "chain: two 2D bf16 cublasGemmEx dispatches "
        "(grad_hidden_states via at::matmul(g, W) OP_N OP_N -- the "
        "verified cublasGemmEx_2D_bf16_OP_N_OP_N family canonical "
        "binding; grad_gate_weight via g.t() @ x OP_T OP_N -- same "
        "family) + quotient-rule normalization backward + topk "
        "straight-through scatter + grad_expert_mask permute/gather/"
        "scatter_add + softmax backward via ATen elementwise / "
        "reduction / scatter ops. Const axes: hidden_size=2048, "
        "num_experts=128, top_k=8. Destination-passing style; "
        "2 gradient outputs.");
}
