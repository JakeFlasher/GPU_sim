#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <string>

// Group-limited top-k expert routing for a Mixture-of-Experts router
// (DeepSeek-style two-stage group selection). Reproduces the reference run()
// op-for-op via pure ATen / cuBLAS dispatch -- no custom device kernels.
//
// Pinned axes from the problem definition:
//   num_experts          = 256
//   num_experts_per_tok  = 8   (final top_k)
//   n_group              = 8
//   topk_group           = 4
//   gating_dim           = 4096
//   experts_per_group    = num_experts / n_group = 32
//
// Per definition.json (insertion order):
//   hidden_states          : [num_tokens, gating_dim=4096]  bf16
//   weight                 : [num_experts=256, gating_dim]   bf16
//   expert_bias            : [num_experts=256]               bf16
//   routed_scaling_factor  : scalar                          fp32 (by value)
//   topk_idx               : [num_tokens, num_experts_per_tok=8] int64 DPS
//   topk_weight            : [num_tokens, num_experts_per_tok=8] fp32 DPS
//
// Algorithm (mirrors the reference precisely, all routing math in fp32):
//   logits             = linear(hidden_states.float(), weight.float())  -> [T,256]
//   scores             = sigmoid(logits)
//   scores_for_routing = scores + expert_bias.float()
//   grouped            = scores_for_routing.view(T, 8, 32)
//   top2_vals,_        = topk(grouped, k=2, dim=-1, largest=true, sorted=false)
//   group_scores       = top2_vals.sum(-1)                              -> [T,8]
//   _,group_idx        = topk(group_scores, k=4, dim=-1, sorted=false)  -> [T,4]
//   group_mask         = zeros_like(group_scores).scatter_(1, group_idx, 1.0)
//   score_mask         = group_mask.unsqueeze(-1)
//                          .expand(T,8,32).reshape(T,256)
//   masked_scores      = scores_for_routing.masked_fill(score_mask == 0,
//                                                        finfo(f32).min)
//   _,topk_idx         = topk(masked_scores, k=8, dim=-1, sorted=false) -> [T,8]
//   selected_scores    = gather(scores, 1, topk_idx)   (original scores, no bias)
//   topk_weight        = selected_scores
//                          / (selected_scores.sum(-1, keepdim) + 1e-20)
//   topk_weight        = topk_weight * routed_scaling_factor
//
// The whole routing pipeline runs in fp32 (the linear casts both operands to
// fp32, matching F.linear on float-upcast inputs), so scores, the group-score
// aggregation, the masking value (the fp32 finfo minimum), and the
// normalization epsilon all stay in fp32. The topk calls match the reference's
// k / dim / largest / sorted arguments exactly so the selected index sets and
// the gather order line up with torch. The two index/weight outputs are cast to
// int64 and fp32 respectively to match the declared output dtypes, then written
// in place (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    const torch::Tensor& expert_bias,
    double routed_scaling_factor,
    torch::Tensor topk_idx,
    torch::Tensor topk_weight
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(weight, "weight");
    check_bf16_cuda(expert_bias, "expert_bias");
    TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
    TORCH_CHECK(topk_idx.dtype() == torch::kInt64,
                "topk_idx must be int64, got ", topk_idx.dtype());
    TORCH_CHECK(topk_weight.is_cuda(), "topk_weight must be a CUDA tensor");
    TORCH_CHECK(topk_weight.dtype() == torch::kFloat32,
                "topk_weight must be float32, got ", topk_weight.dtype());
    TORCH_CHECK(hidden_states.dim() == 2,
                "hidden_states must be 2D [num_tokens, gating_dim], got ",
                hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumExperts = 256;
    constexpr int64_t kTopK = 8;
    constexpr int64_t kNumGroups = 8;
    constexpr int64_t kTopkGroup = 4;
    constexpr int64_t kExpertsPerGroup = kNumExperts / kNumGroups;  // 32
    constexpr double kNormEps = 1e-20;

    const int64_t num_tokens = hidden_states.size(0);

    // Routing logits in fp32: linear upcasts both operands to fp32 first.
    auto logits = at::linear(hidden_states.to(torch::kFloat32),
                             weight.to(torch::kFloat32));  // [T, 256]

    // Sigmoid routing scores; bias is added on a fp32 copy for the routing
    // path only (the original scores are reused for the final weights).
    auto scores = at::sigmoid(logits);                          // [T, 256]
    auto scores_for_routing = scores + expert_bias.to(torch::kFloat32);

    // Per-group top-2 sum -> group scores [T, 8].
    auto grouped = scores_for_routing.view({num_tokens, kNumGroups, kExpertsPerGroup});
    auto top2 = at::topk(grouped, /*k=*/2, /*dim=*/-1,
                         /*largest=*/true, /*sorted=*/false);
    auto group_scores = std::get<0>(top2).sum(/*dim=*/-1);      // [T, 8]

    // Top-4 groups; scatter a {0,1} group mask over the selected groups.
    auto group_topk = at::topk(group_scores, /*k=*/kTopkGroup, /*dim=*/-1,
                               /*largest=*/true, /*sorted=*/false);
    auto group_idx = std::get<1>(group_topk);                   // [T, 4]
    auto group_mask = at::zeros_like(group_scores);
    group_mask.scatter_(/*dim=*/1, group_idx, 1.0);            // [T, 8]

    // Expand the group mask to expert granularity and mask out non-selected
    // experts with the fp32 finfo minimum (matching torch.finfo(f32).min).
    auto score_mask = group_mask.unsqueeze(-1)
                          .expand({num_tokens, kNumGroups, kExpertsPerGroup})
                          .reshape({num_tokens, kNumExperts});  // [T, 256]
    const double neg_fill =
        static_cast<double>(std::numeric_limits<float>::lowest());
    auto masked_scores = scores_for_routing.masked_fill(score_mask == 0, neg_fill);

    // Top-8 experts over the masked scores -> selected indices.
    auto expert_topk = at::topk(masked_scores, /*k=*/kTopK, /*dim=*/-1,
                                /*largest=*/true, /*sorted=*/false);
    auto idx = std::get<1>(expert_topk);                        // [T, 8] int64

    // Gather the original (bias-free) scores, normalize to sum 1, then scale.
    auto selected = at::gather(scores, /*dim=*/1, idx);         // [T, 8]
    auto weight_out = selected / (selected.sum(/*dim=*/-1, /*keepdim=*/true) + kNormEps);
    weight_out = weight_out * routed_scaling_factor;

    // Destination-passing-style writes; cast to the declared output dtypes.
    TORCH_CHECK(topk_idx.sizes() == idx.sizes(),
                "topk_idx shape ", topk_idx.sizes(),
                " does not match computed ", idx.sizes());
    TORCH_CHECK(topk_weight.sizes() == weight_out.sizes(),
                "topk_weight shape ", topk_weight.sizes(),
                " does not match computed ", weight_out.sizes());
    topk_idx.copy_(idx.to(torch::kInt64));
    topk_weight.copy_(weight_out.to(torch::kFloat32));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Group-limited top-k MoE routing. Pure ATen / cuBLAS dispatch: fp32 "
          "linear routing logits, sigmoid scores, expert-bias add, per-group "
          "top-2 sum into group scores, top-4 group selection with a scattered "
          "{0,1} group mask, expert-level mask-fill with the fp32 finfo minimum, "
          "top-8 expert selection, gather of the bias-free scores, sum-to-1 "
          "normalization (1e-20 epsilon) and routed_scaling_factor scaling. "
          "Outputs int64 indices and fp32 weights; destination-passing style.");
}
