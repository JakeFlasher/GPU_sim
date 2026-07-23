#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Backward pass for MoE expert load balancing and token capacity.
//
// Reference (problem_042_moe_expert_load_balancing_and_token_capacity_backward):
//
//     batch_seq_len = topk_idx.shape[0]
//     n_routed_experts = 256          # HARDCODED const axis in run()
//     num_experts_per_tok = 8         # HARDCODED const axis in run()
//
//     grad_loss_val = grad_load_balance_loss.item()
//     is_training   = training.item()
//
//     grad_expert_mask_out = grad_expert_mask.clone()
//     grad_tpe_accumulated = grad_tokens_per_expert.clone()
//
//     if is_training:
//         uniform_prob   = 1.0 / n_routed_experts
//         grad_from_loss = grad_loss_val * n_routed_experts * uniform_prob \
//                          / (batch_seq_len * num_experts_per_tok)
//         grad_tpe_accumulated = grad_tpe_accumulated + grad_from_loss
//
//     grad_expert_mask_from_sum = grad_tpe_accumulated.unsqueeze(0) \
//                                     .expand(batch_seq_len, n_routed_experts).float()
//     grad_expert_mask_out = grad_expert_mask_out + grad_expert_mask_from_sum
//     return grad_expert_mask_out
//
// Per definition.json (insertion order):
//   grad_tokens_per_expert  : [n_routed_experts]                  fp32
//   grad_expert_mask        : [batch_seq_len, n_routed_experts]   fp32
//   grad_load_balance_loss  : [1]                                 fp32  (1-elem tensor "scalar")
//   topk_idx                : [batch_seq_len, num_experts_per_tok] int64 (UNUSED in run())
//   expert_mask             : [batch_seq_len, n_routed_experts]   int64 (UNUSED in run())
//   tokens_per_expert       : [n_routed_experts]                  int32 (UNUSED in run())
//   training                : [1]                                 bool  (1-elem tensor "scalar")
//   grad_expert_mask_out    : [batch_seq_len, n_routed_experts]   fp32  (DPS, pre-alloc)
//
// Subtleties faithfully reproduced:
//   * topk_idx / expert_mask / tokens_per_expert are accepted but NEVER read by
//     the reference run() (discrete indices / shape-only saved-for-backward
//     tensors with no gradient flow). We mirror that exactly: they are
//     light-validated for presence only and otherwise unused.
//   * The reference reads batch_seq_len from topk_idx.shape[0]. That value is
//     identical to grad_expert_mask.size(0) by construction (both == the
//     batch_seq_len axis). We derive it from grad_expert_mask, the tensor that
//     actually carries the value into the arithmetic.
//   * n_routed_experts / num_experts_per_tok are HARDCODED constants in the
//     reference (256 / 8), independent of any runtime shape. We bake the same
//     constants in so the load-balance arithmetic is bit-identical. (Note
//     n_routed_experts * (1.0/n_routed_experts) == 1.0 algebraically, but we
//     keep the reference's explicit factored form so FP rounding matches.)
//   * The `training` branch: when training.item() is False the grad-from-loss
//     term is dropped entirely; grad_tpe_accumulated stays == grad_tokens_per_expert.
//
// Implementation: pure ATen tensor ops -- NO custom CUDA kernels. Same dtypes
// and broadcast semantics as the reference; output written in place into the
// pre-allocated destination (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_tokens_per_expert,
    const torch::Tensor& grad_expert_mask,
    const torch::Tensor& grad_load_balance_loss,
    const torch::Tensor& topk_idx,
    const torch::Tensor& expert_mask,
    const torch::Tensor& tokens_per_expert,
    const torch::Tensor& training,
    torch::Tensor grad_expert_mask_out
) {
    check_fp32_cuda(grad_tokens_per_expert, "grad_tokens_per_expert");
    check_fp32_cuda(grad_expert_mask, "grad_expert_mask");
    check_fp32_cuda(grad_load_balance_loss, "grad_load_balance_loss");
    check_fp32_cuda(grad_expert_mask_out, "grad_expert_mask_out");
    TORCH_CHECK(grad_expert_mask.dim() == 2,
                "grad_expert_mask must be 2D [batch_seq_len, n_routed_experts], got ",
                grad_expert_mask.dim(), "D");
    TORCH_CHECK(grad_tokens_per_expert.dim() == 1,
                "grad_tokens_per_expert must be 1D [n_routed_experts], got ",
                grad_tokens_per_expert.dim(), "D");
    TORCH_CHECK(grad_load_balance_loss.numel() == 1,
                "grad_load_balance_loss must be a 1-element tensor, got numel ",
                grad_load_balance_loss.numel());
    TORCH_CHECK(training.numel() == 1,
                "training must be a 1-element tensor, got numel ", training.numel());
    TORCH_CHECK(training.dtype() == torch::kBool,
                "training must be bool, got ", training.dtype());

    const c10::cuda::CUDAGuard device_guard(grad_expert_mask.device());

    // topk_idx / expert_mask / tokens_per_expert are intentionally unused -- the
    // reference run() never reads them (discrete indices / shape-only saved
    // tensors with no gradient contribution). Cast to void to silence
    // unused-parameter warnings without altering semantics.
    (void)topk_idx;
    (void)expert_mask;
    (void)tokens_per_expert;

    // batch_seq_len: reference uses topk_idx.shape[0]; identical to
    // grad_expert_mask.size(0) by construction. Use the gradient tensor.
    const int64_t batch_seq_len = grad_expert_mask.size(0);
    constexpr int64_t kNRoutedExperts = 256;   // hardcoded const axis (definition.json const value)
    constexpr int64_t kNumExpertsPerTok = 8;   // hardcoded const axis (definition.json const value)

    // Read the 1-element tensor "scalars" (matches grad_load_balance_loss.item()
    // / training.item()). .item<...>() forces a device->host sync on those
    // scalars, exactly as the reference .item() calls do.
    const double grad_loss_val = grad_load_balance_loss.item<double>();
    const bool is_training = training.item<bool>();

    // grad_tpe_accumulated = grad_tokens_per_expert.clone()  (+ grad_from_loss if training)
    torch::Tensor grad_tpe_accumulated = grad_tokens_per_expert.clone();
    if (is_training) {
        const double uniform_prob = 1.0 / static_cast<double>(kNRoutedExperts);
        const double grad_from_loss =
            grad_loss_val * static_cast<double>(kNRoutedExperts) * uniform_prob /
            (static_cast<double>(batch_seq_len) * static_cast<double>(kNumExpertsPerTok));
        // fp32-tensor + Python-float scalar broadcast: ATen keeps the fp32
        // result dtype, matching `grad_tpe_accumulated + grad_from_loss`.
        grad_tpe_accumulated = grad_tpe_accumulated + grad_from_loss;
    }

    // grad_expert_mask_out = grad_expert_mask.clone()
    //   + grad_tpe_accumulated.unsqueeze(0).expand(batch_seq_len, n_routed_experts).float()
    torch::Tensor grad_expert_mask_from_sum =
        grad_tpe_accumulated.unsqueeze(0)
            .expand({batch_seq_len, kNRoutedExperts})
            .to(torch::kFloat32);
    torch::Tensor result = grad_expert_mask.clone() + grad_expert_mask_from_sum;

    TORCH_CHECK(result.sizes() == grad_expert_mask_out.sizes(),
                "computed grad_expert_mask_out shape ", result.sizes(),
                " does not match pre-allocated output ", grad_expert_mask_out.sizes());
    grad_expert_mask_out.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "MoE expert load-balancing backward via ATen (clone + broadcast-add); "
          "fp32; destination-passing style. topk_idx/expert_mask/tokens_per_expert "
          "are unused (mirrors the reference).");
}
