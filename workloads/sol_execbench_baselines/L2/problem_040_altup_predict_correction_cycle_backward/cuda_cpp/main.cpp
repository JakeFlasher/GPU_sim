#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

// Backward pass for the AltUp predict-correct cycle.
//
// Reference (problem_040_altup_predict_correction_cycle_backward): the analytic
// backward written out by hand as explicit tensor algebra (no torch.autograd).
// The reference recomputes the forward of both the predict and the correct
// step in float32, then walks the gradients back through them. This port
// reproduces that op-for-op.
//
// Precision profile (matched exactly): every input is loaded and the whole
// recompute / backward runs in float32 (the reference promotes each bf16 input
// via .float() at its first use). Two outputs -- grad_hidden_states and
// grad_activated -- are cast back to bfloat16 at the very end (the reference's
// .to(torch.bfloat16)); the four weight gradients stay float32.
//
// Forward chain the backward inverts (per the active altup index):
//   predict step:
//     x        = hidden_states[idx].float()
//     rstd     = rsqrt(mean(x^2, -1) + eps)
//     normed   = (x * rstd) * norm_weight
//     scaled   = normed * router_scale            (router_scale = 1/hidden_size)
//     routed   = linear(scaled, router_weight)
//     modal    = tanh(routed)
//     coefs    = linear(modal, prediction_coef_weight)
//                  reshaped to [B,S,n,n], permuted (0,1,3,2)
//     pred     = matmul(h.permute(1,2,3,0), coefs).permute(3,0,1,2) + hidden
//   correct step:
//     (same RMSNorm/router/tanh chain on `activated`)
//     coefs_c  = linear(modal_c, correction_coef_weight) + 1
//     innov    = activated - pred[idx], broadcast over n
//     corrected= innov * coefs_c.permute(2,0,1).unsqueeze(-1) + pred
//
// Per definition.json (insertion order):
//   inputs:
//     grad_corrected         : [n, B, S, hidden_size]   bf16
//     hidden_states          : [n, B, S, hidden_size]   bf16
//     activated              : [B, S, hidden_size]       bf16
//     prediction_coef_weight : [n*n, n]                  bf16
//     correction_coef_weight : [n, n]                    bf16
//     router_weight          : [n, hidden_size]          bf16
//     norm_weight            : [hidden_size]             bf16
//     altup_active_idx       : scalar int32 -> int64_t
//     rms_norm_eps           : scalar float32 -> double
//   outputs (DPS pre-alloc, written in place; insertion order):
//     grad_hidden_states         : [n, B, S, hidden_size]  bf16
//     grad_activated             : [B, S, hidden_size]      bf16
//     grad_prediction_coef_weight: [n*n, n]                 fp32
//     grad_correction_coef_weight: [n, n]                   fp32
//     grad_router_weight         : [n, hidden_size]         fp32
//     grad_norm_weight           : [hidden_size]            fp32
//
// Implementation: pure ATen / cuBLAS dispatch -- no custom CUDA kernels. The
// linear / matmul calls route through the same cuBLAS GEMM the reference's
// F.linear / torch.matmul use; the RMSNorm backward algebra, the tanh
// derivative, and every reshape / permute / sum orientation match the
// reference exactly. The active-index updates use static at::select index ops,
// not loops.

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_corrected,
    const torch::Tensor& hidden_states,
    const torch::Tensor& activated,
    const torch::Tensor& prediction_coef_weight,
    const torch::Tensor& correction_coef_weight,
    const torch::Tensor& router_weight,
    const torch::Tensor& norm_weight,
    int64_t altup_active_idx,
    double rms_norm_eps,
    torch::Tensor grad_hidden_states,
    torch::Tensor grad_activated,
    torch::Tensor grad_prediction_coef_weight,
    torch::Tensor grad_correction_coef_weight,
    torch::Tensor grad_router_weight,
    torch::Tensor grad_norm_weight
) {
    check_bf16_cuda(grad_corrected, "grad_corrected");
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(activated, "activated");
    check_bf16_cuda(prediction_coef_weight, "prediction_coef_weight");
    check_bf16_cuda(correction_coef_weight, "correction_coef_weight");
    check_bf16_cuda(router_weight, "router_weight");
    check_bf16_cuda(norm_weight, "norm_weight");

    TORCH_CHECK(hidden_states.dim() == 4,
                "hidden_states must be 4D [n, B, S, hidden_size], got ",
                hidden_states.dim(), "D");
    TORCH_CHECK(activated.dim() == 3,
                "activated must be 3D [B, S, hidden_size], got ", activated.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t altup_num_inputs = hidden_states.size(0);
    const int64_t batch_size = hidden_states.size(1);
    const int64_t seq_len = hidden_states.size(2);
    const int64_t hidden_size = hidden_states.size(3);
    const double router_scale = 1.0 / static_cast<double>(hidden_size);
    const int64_t idx = altup_active_idx;

    TORCH_CHECK(idx >= 0 && idx < altup_num_inputs,
                "altup_active_idx ", idx, " out of range [0, ", altup_num_inputs, ")");

    // Everything is promoted to float32, mirroring the reference's per-input
    // .float() casts.
    auto grad_corrected_f = grad_corrected.to(torch::kFloat32);
    auto hidden_f = hidden_states.to(torch::kFloat32);
    auto activated_f = activated.to(torch::kFloat32);
    auto pred_w_f = prediction_coef_weight.to(torch::kFloat32);
    auto corr_w_f = correction_coef_weight.to(torch::kFloat32);
    auto router_w_f = router_weight.to(torch::kFloat32);
    auto norm_w_f = norm_weight.to(torch::kFloat32);

    // ===== Forward recomputation: predict step =====
    auto x_predict = hidden_f.select(/*dim=*/0, idx);  // [B, S, hidden_size]
    auto variance_predict = x_predict.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto rstd_predict = at::rsqrt(variance_predict + rms_norm_eps);
    auto normalized_predict = x_predict * rstd_predict;
    auto normed_predict = normalized_predict * norm_w_f;
    auto scaled_predict = normed_predict * router_scale;
    auto routed_predict = at::linear(scaled_predict, router_w_f);
    auto modalities_predict = at::tanh(routed_predict);

    auto all_coefs_flat = at::linear(modalities_predict, pred_w_f);
    auto all_coefs = all_coefs_flat.reshape(
        {batch_size, seq_len, altup_num_inputs, altup_num_inputs});
    all_coefs = all_coefs.permute({0, 1, 3, 2});

    auto h_permuted = hidden_f.permute({1, 2, 3, 0});
    auto predictions_before_residual = at::matmul(h_permuted, all_coefs);
    auto predictions_permuted = predictions_before_residual.permute({3, 0, 1, 2});
    auto predictions = predictions_permuted + hidden_f;

    // ===== Forward recomputation: correct step =====
    auto x_correct = activated_f;
    auto variance_correct = x_correct.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto rstd_correct = at::rsqrt(variance_correct + rms_norm_eps);
    auto normalized_correct = x_correct * rstd_correct;
    auto normed_correct = normalized_correct * norm_w_f;
    auto scaled_correct = normed_correct * router_scale;
    auto routed_correct = at::linear(scaled_correct, router_w_f);
    auto modalities_correct = at::tanh(routed_correct);

    auto innovation = activated_f - predictions.select(/*dim=*/0, idx);
    auto innovation_repeated =
        innovation.unsqueeze(0).expand({altup_num_inputs, -1, -1, -1});

    auto all_coefs_correct = at::linear(modalities_correct, corr_w_f) + 1.0;
    auto all_coefs_expanded =
        all_coefs_correct.permute({2, 0, 1}).unsqueeze(-1);

    // ===== Backward: correct step =====
    // Gradient through the residual connection.
    auto grad_predictions = grad_corrected_f.clone();

    // Gradient through corrected = innovation_repeated * all_coefs_expanded + predictions.
    auto grad_innovation_repeated = grad_corrected_f * all_coefs_expanded;
    auto grad_all_coefs_expanded =
        (grad_corrected_f * innovation_repeated).sum(/*dim=*/-1, /*keepdim=*/true);

    // Gradient through unsqueeze + permute of all_coefs_correct.
    auto grad_all_coefs_correct =
        grad_all_coefs_expanded.squeeze(-1).permute({1, 2, 0});

    // Gradient through the +1.0 (identity) then the correction linear projection.
    auto grad_correction_coef_weight_result = at::matmul(
        grad_all_coefs_correct.reshape({-1, altup_num_inputs}).t(),
        modalities_correct.reshape({-1, altup_num_inputs}));

    auto grad_modalities_correct =
        at::matmul(grad_all_coefs_correct, corr_w_f);

    // Gradient through the broadcast of innovation (sum over repeated dim).
    auto grad_innovation = grad_innovation_repeated.sum(/*dim=*/0);

    // Gradient through innovation = activated - predictions[idx]. The reference's
    // in-place grad_predictions[idx] -= grad_innovation is a static select-at-idx
    // subtraction on the active row (not a loop).
    auto grad_activated_accum = grad_innovation.clone();
    grad_predictions.select(/*dim=*/0, idx).sub_(grad_innovation);

    // Gradient through the tanh router modalities (correct).
    auto grad_routed_correct =
        grad_modalities_correct * (1.0 - modalities_correct.pow(2));

    auto grad_router_weight_correct = at::matmul(
        grad_routed_correct.reshape({-1, altup_num_inputs}).t(),
        scaled_correct.reshape({-1, hidden_size}));

    auto grad_scaled_correct = at::matmul(grad_routed_correct, router_w_f);
    auto grad_normed_correct = grad_scaled_correct * router_scale;

    auto grad_norm_weight_correct =
        (grad_normed_correct * normalized_correct).sum(/*dim=*/std::vector<int64_t>{0, 1});

    auto grad_normalized_correct = grad_normed_correct * norm_w_f;
    auto mean_grad_normalized_x_correct =
        (grad_normalized_correct * x_correct).mean(/*dim=*/-1, /*keepdim=*/true);
    auto grad_activated_from_router =
        grad_normalized_correct * rstd_correct -
        x_correct * rstd_correct.pow(3) * mean_grad_normalized_x_correct;
    grad_activated_accum = grad_activated_accum + grad_activated_from_router;

    // ===== Backward: predict step =====
    // Gradient through the residual connection.
    auto grad_hidden_states_accum = grad_predictions.clone();

    auto grad_predictions_permuted = grad_predictions.permute({1, 2, 3, 0});

    // Gradient through predictions = matmul(h_permuted, all_coefs).
    auto grad_h_permuted =
        at::matmul(grad_predictions_permuted, all_coefs.transpose(-2, -1));
    auto grad_all_coefs_matmul =
        at::matmul(h_permuted.transpose(-2, -1), grad_predictions_permuted);

    // Gradient through the permute of hidden_states.
    grad_hidden_states_accum =
        grad_hidden_states_accum + grad_h_permuted.permute({3, 0, 1, 2});

    // Gradient through the transpose of all_coefs.
    auto grad_all_coefs_before_transpose =
        grad_all_coefs_matmul.permute({0, 1, 3, 2});

    // Gradient through the reshape.
    auto grad_all_coefs_flat = grad_all_coefs_before_transpose.reshape(
        {batch_size, seq_len, altup_num_inputs * altup_num_inputs});

    // Gradient through the prediction linear projection.
    auto grad_prediction_coef_weight_result = at::matmul(
        grad_all_coefs_flat.reshape({-1, altup_num_inputs * altup_num_inputs}).t(),
        modalities_predict.reshape({-1, altup_num_inputs}));

    auto grad_modalities_predict = at::matmul(grad_all_coefs_flat, pred_w_f);

    // Gradient through the tanh router modalities (predict).
    auto grad_routed_predict =
        grad_modalities_predict * (1.0 - modalities_predict.pow(2));

    auto grad_router_weight_predict = at::matmul(
        grad_routed_predict.reshape({-1, altup_num_inputs}).t(),
        scaled_predict.reshape({-1, hidden_size}));

    auto grad_scaled_predict = at::matmul(grad_routed_predict, router_w_f);
    auto grad_normed_predict = grad_scaled_predict * router_scale;

    auto grad_norm_weight_predict =
        (grad_normed_predict * normalized_predict).sum(/*dim=*/std::vector<int64_t>{0, 1});

    auto grad_normalized_predict = grad_normed_predict * norm_w_f;
    auto mean_grad_normalized_x_predict =
        (grad_normalized_predict * x_predict).mean(/*dim=*/-1, /*keepdim=*/true);
    auto grad_active_input =
        grad_normalized_predict * rstd_predict -
        x_predict * rstd_predict.pow(3) * mean_grad_normalized_x_predict;

    // The reference's in-place grad_hidden_states[idx] += grad_active_input is a
    // static select-at-idx addition on the active row (not a loop).
    grad_hidden_states_accum.select(/*dim=*/0, idx).add_(grad_active_input);

    // Combine the predict and correct contributions to the shared parameters.
    auto grad_router_weight_result =
        grad_router_weight_predict + grad_router_weight_correct;
    auto grad_norm_weight_result =
        grad_norm_weight_predict + grad_norm_weight_correct;

    // Two activation/grad outputs are cast back to bfloat16; weight grads stay fp32.
    auto grad_hidden_states_bf16 = grad_hidden_states_accum.to(torch::kBFloat16);
    auto grad_activated_bf16 = grad_activated_accum.to(torch::kBFloat16);

    // Destination-passing-style writes (definition.json output order).
    TORCH_CHECK(grad_hidden_states_bf16.sizes() == grad_hidden_states.sizes(),
                "grad_hidden_states shape ", grad_hidden_states_bf16.sizes(),
                " does not match pre-allocated ", grad_hidden_states.sizes());
    TORCH_CHECK(grad_activated_bf16.sizes() == grad_activated.sizes(),
                "grad_activated shape ", grad_activated_bf16.sizes(),
                " does not match pre-allocated ", grad_activated.sizes());
    TORCH_CHECK(grad_prediction_coef_weight_result.sizes() == grad_prediction_coef_weight.sizes(),
                "grad_prediction_coef_weight shape ", grad_prediction_coef_weight_result.sizes(),
                " does not match pre-allocated ", grad_prediction_coef_weight.sizes());
    TORCH_CHECK(grad_correction_coef_weight_result.sizes() == grad_correction_coef_weight.sizes(),
                "grad_correction_coef_weight shape ", grad_correction_coef_weight_result.sizes(),
                " does not match pre-allocated ", grad_correction_coef_weight.sizes());
    TORCH_CHECK(grad_router_weight_result.sizes() == grad_router_weight.sizes(),
                "grad_router_weight shape ", grad_router_weight_result.sizes(),
                " does not match pre-allocated ", grad_router_weight.sizes());
    TORCH_CHECK(grad_norm_weight_result.sizes() == grad_norm_weight.sizes(),
                "grad_norm_weight shape ", grad_norm_weight_result.sizes(),
                " does not match pre-allocated ", grad_norm_weight.sizes());

    grad_hidden_states.copy_(grad_hidden_states_bf16);
    grad_activated.copy_(grad_activated_bf16);
    grad_prediction_coef_weight.copy_(grad_prediction_coef_weight_result);
    grad_correction_coef_weight.copy_(grad_correction_coef_weight_result);
    grad_router_weight.copy_(grad_router_weight_result);
    grad_norm_weight.copy_(grad_norm_weight_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for the AltUp predict-correct cycle. Pure ATen / cuBLAS "
        "dispatch: forward recompute (RMSNorm + router + tanh + coefficient "
        "linears + coefficient matmul) of both predict and correct steps in "
        "float32, then the analytic backward reproduced op-for-op from the "
        "reference. grad_hidden_states / grad_activated cast back to bfloat16, "
        "the four weight gradients stay float32. Active-index updates via "
        "at::select. Destination-passing style (6 outputs).");
}
