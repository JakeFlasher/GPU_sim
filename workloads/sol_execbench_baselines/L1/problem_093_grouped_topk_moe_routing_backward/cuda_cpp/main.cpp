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

static void check_int64_contiguous_cuda(
    const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kInt64,
                name, " must be int64, got ", tensor.dtype());
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Destination-passing-style entry for the grouped top-k MoE routing
// backward. Per definition.json:
//   grad_topk_weights         : [N, top_k=8]                bf16
//   hidden_states             : [N, hidden_size=5120]       bf16
//   weight                    : [n_routed_experts=160,
//                                hidden_size=5120]          bf16
//   scores                    : [N, n_routed_experts]       bf16
//   topk_indices              : [N, top_k]                  int64
//   topk_weights              : [N, top_k]                  bf16
//   topk_weights_normalized   : [N, top_k]                  bf16
//   denominator               : [N, 1]                      bf16
//   routed_scaling_factor     : scalar bf16
//   grad_hidden_states        : [N, hidden_size]            bf16 DPS
//   grad_weight               : [n_routed_experts,
//                                hidden_size]               bf16 DPS
//
// Algorithm (mirrors the reference exactly):
//   grad_topk_weights_normalized = grad_topk_weights * scaling
//   grad_sum = (grad_topk_weights_normalized *
//               topk_weights_normalized).sum(-1, keepdim=true)
//   grad_topk_weights_unnorm = (grad_topk_weights_normalized
//                               - grad_sum) / denominator
//   grad_scores = zeros(N, n_routed_experts)
//   grad_scores.scatter_add_(1, topk_indices, grad_topk_weights_unnorm)
//   grad_router_logits = grad_scores * scores * (1 - scores)
//   grad_hidden_states = grad_router_logits @ weight   (OP_N OP_N)
//   grad_weight = grad_router_logits.t() @ hidden_states (OP_T OP_N)
//
// Dispatch surfaces:
//   * Two 2D bf16 sgemm dispatches (grad_hidden_states + grad_weight)
//     bind cublasGemmEx OP_N OP_N (verified family) and OP_T OP_N
//     (same family's transposed-A variant).
//   * Sigmoid backward, gather/scatter backward, quotient-rule
//     normalization backward run through ATen elementwise /
//     reduction / scatter_add ops.
void run(
    torch::Tensor grad_topk_weights,
    torch::Tensor hidden_states,
    torch::Tensor weight,
    torch::Tensor scores,
    torch::Tensor topk_indices,
    torch::Tensor topk_weights,
    torch::Tensor topk_weights_normalized,
    torch::Tensor denominator,
    double routed_scaling_factor,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_weight
) {
    check_bf16_contiguous_cuda(grad_topk_weights, "grad_topk_weights");
    check_bf16_contiguous_cuda(hidden_states, "hidden_states");
    check_bf16_contiguous_cuda(weight, "weight");
    check_bf16_contiguous_cuda(scores, "scores");
    check_int64_contiguous_cuda(topk_indices, "topk_indices");
    check_bf16_contiguous_cuda(topk_weights, "topk_weights");
    check_bf16_contiguous_cuda(topk_weights_normalized,
                               "topk_weights_normalized");
    check_bf16_contiguous_cuda(denominator, "denominator");
    check_bf16_contiguous_cuda(grad_hidden_states, "grad_hidden_states");
    check_bf16_contiguous_cuda(grad_weight, "grad_weight");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kHiddenSize = 5120;
    constexpr int64_t kNumRoutedExperts = 160;
    constexpr int64_t kTopK = 8;

    TORCH_CHECK(hidden_states.dim() == 2 &&
                hidden_states.size(1) == kHiddenSize,
                "hidden_states must be [N, hidden_size=5120]");
    const int64_t N = hidden_states.size(0);

    TORCH_CHECK(grad_topk_weights.dim() == 2 &&
                grad_topk_weights.size(0) == N &&
                grad_topk_weights.size(1) == kTopK,
                "grad_topk_weights must be [N, top_k=8]");
    TORCH_CHECK(weight.dim() == 2 &&
                weight.size(0) == kNumRoutedExperts &&
                weight.size(1) == kHiddenSize,
                "weight must be [n_routed_experts=160, hidden_size=5120]");
    TORCH_CHECK(scores.dim() == 2 &&
                scores.size(0) == N &&
                scores.size(1) == kNumRoutedExperts,
                "scores must be [N, n_routed_experts]");
    TORCH_CHECK(topk_indices.dim() == 2 &&
                topk_indices.size(0) == N &&
                topk_indices.size(1) == kTopK,
                "topk_indices must be [N, top_k]");
    TORCH_CHECK(topk_weights.dim() == 2 &&
                topk_weights.size(0) == N &&
                topk_weights.size(1) == kTopK,
                "topk_weights must be [N, top_k]");
    TORCH_CHECK(topk_weights_normalized.dim() == 2 &&
                topk_weights_normalized.size(0) == N &&
                topk_weights_normalized.size(1) == kTopK,
                "topk_weights_normalized must be [N, top_k]");
    TORCH_CHECK(denominator.dim() == 2 &&
                denominator.size(0) == N &&
                denominator.size(1) == 1,
                "denominator must be [N, 1]");
    TORCH_CHECK(grad_hidden_states.dim() == 2 &&
                grad_hidden_states.size(0) == N &&
                grad_hidden_states.size(1) == kHiddenSize,
                "grad_hidden_states must be [N, hidden_size]");
    TORCH_CHECK(grad_weight.dim() == 2 &&
                grad_weight.size(0) == kNumRoutedExperts &&
                grad_weight.size(1) == kHiddenSize,
                "grad_weight must be [n_routed_experts, hidden_size]");

    // ----------------------------------------------------------------
    // Gradient through scaling:
    //   topk_weights_scaled = topk_weights_normalized * scaling
    //   grad_normalized = grad_topk_weights * scaling
    // ----------------------------------------------------------------
    auto grad_topk_weights_normalized = grad_topk_weights * routed_scaling_factor;

    // ----------------------------------------------------------------
    // Gradient through per-row normalization (quotient rule):
    //   topk_weights_normalized = topk_weights / denominator
    //   grad_topk_weights_unnorm =
    //     (grad_normalized - sum(grad_normalized *
    //                            topk_weights_normalized, -1))
    //     / denominator
    // ----------------------------------------------------------------
    auto grad_sum = (grad_topk_weights_normalized
                     * topk_weights_normalized)
                        .sum(/*dim=*/-1, /*keepdim=*/true);
    auto grad_topk_weights_unnorm =
        (grad_topk_weights_normalized - grad_sum) / denominator;

    // ----------------------------------------------------------------
    // Gradient through gather operation:
    //   topk_weights = scores.gather(1, topk_indices)
    //   backward = scatter_add of grad_topk_weights_unnorm into
    //   grad_scores at the topk_indices positions.
    // ----------------------------------------------------------------
    auto grad_scores = at::zeros(
        {N, kNumRoutedExperts},
        grad_topk_weights_unnorm.options());
    grad_scores.scatter_add_(/*dim=*/1, topk_indices,
                             grad_topk_weights_unnorm);

    // ----------------------------------------------------------------
    // Gradient through sigmoid:
    //   scores = sigmoid(router_logits)
    //   d_sigmoid/d_x = scores * (1 - scores)
    // ----------------------------------------------------------------
    auto grad_router_logits = grad_scores * scores * (1.0 - scores);

    // ----------------------------------------------------------------
    // Gradient through linear projection:
    //   router_logits = hidden_states @ weight.T
    //   grad_hidden_states = grad_router_logits @ weight
    //                        (cublasGemmEx 2D bf16 OP_N OP_N -- the
    //                         verified cublasGemmEx_2D_bf16_OP_N_OP_N
    //                         family canonical binding).
    //   grad_weight = grad_router_logits.T @ hidden_states
    //                 (cublasGemmEx 2D bf16 OP_T OP_N -- same
    //                  verified family).
    // ----------------------------------------------------------------
    grad_hidden_states.copy_(at::matmul(grad_router_logits, weight));
    grad_weight.copy_(
        at::matmul(grad_router_logits.transpose(0, 1), hidden_states));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Grouped top-k MoE routing backward. Pure ATen / cuBLAS "
        "dispatch chain: two 2D bf16 cublasGemmEx dispatches "
        "(grad_hidden_states via at::matmul(g, W) OP_N OP_N -- "
        "verified family; grad_weight via g.t() @ x OP_T OP_N -- "
        "same family) + sigmoid backward + gather/scatter_add + "
        "quotient-rule normalization backward via ATen elementwise "
        "/ reduction ops. Const axes: hidden_size=5120, "
        "n_routed_experts=160, top_k=8. Destination-passing style; "
        "2 gradient outputs.");
}
