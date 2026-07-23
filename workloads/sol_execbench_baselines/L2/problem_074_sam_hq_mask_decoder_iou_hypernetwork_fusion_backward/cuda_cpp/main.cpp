#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <array>
#include <string>

// Backward pass for the SAM-HQ mask decoder IoU prediction and hypernetwork
// weight generation. Computes gradients for five MLPs: one IoU prediction MLP
// (256 -> 1024 -> 1024 -> 4) and four parallel hypernetwork MLPs
// (256 -> 256 -> 256 -> 32). This is the backward of the L1 086 forward and
// shares its MLP structure.
//
// Reference (problem_074_sam_hq_mask_decoder_iou_hypernetwork_fusion_backward):
// the analytic backward written out by hand as explicit tensor algebra (NO
// torch.autograd). Everything stays in the native float32 dtype the reference
// uses, so there are no cast points to track. Reproduced op-for-op below.
//
// Forward chain each MLP inverts, where a layer is linear(in, W, b) =
// in @ W.T + b with W in [out, in] layout:
//   IoU:    h1 = relu(linear(iou_token_out, iou_proj_in_*))     # 256 -> 1024
//           h2 = relu(linear(h1, iou_hidden_*))                 # 1024 -> 1024
//           iou_scores = linear(h2, iou_proj_out_*)             # 1024 -> 4
//   hyper i (per mask token slice mask_tokens_out[:, :, i, :]):
//           h1 = relu(linear(token_i, hyper_proj_in_*[i]))      # 256 -> 256
//           h2 = relu(linear(h1, hyper_hidden_*[i]))            # 256 -> 256
//           w_i = linear(h2, hyper_proj_out_*[i])               # 256 -> 32
//   The four w_i are stacked on axis 2 into hyper_weights.
//
// Linear backward, with grad_out the upstream gradient and "in" the layer
// input, both flattened to 2D over the batch/point dims:
//   grad_W  = grad_out.T @ in        (sums over the flattened B*P rows)
//   grad_b  = grad_out.sum(dim=0)
//   grad_in = grad_out @ W
// ReLU backward uses the saved pre-ReLU activation: grad masked where the
// pre-ReLU value is > 0 (matching the reference's (pre_relu > 0).float()).
//
// Per definition.json the four hyper experts arrive as batched
// [num_mask_tokens, ...] weight / bias tensors and the per-expert post/pre
// activations arrive as the eight hyper_hidden{1,2}{,_relu}_{0..3} tensors;
// the four per-expert grads are written back into the [num_mask_tokens, ...]
// output slices. Grad masks are read off the saved pre-ReLU tensors
// (iou_hidden1, iou_hidden2, hyper_hidden1_i, hyper_hidden2_i).
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// .t().mm() / .mm() calls route through the same cuBLAS GEMM the reference's
// torch ops use; every reshape, matmul orientation, bias-sum axis, and ReLU
// mask matches the reference exactly. Native float32 throughout. The four
// hypernetwork experts run in a host loop, exactly as the reference's
// for mask_idx in range(4). Destination-passing style (14 outputs).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_iou_scores,
    const torch::Tensor& grad_hyper_weights,
    const torch::Tensor& iou_token_out,
    const torch::Tensor& mask_tokens_out,
    const torch::Tensor& iou_proj_in_weight,
    const torch::Tensor& iou_proj_in_bias,
    const torch::Tensor& iou_hidden_weight,
    const torch::Tensor& iou_hidden_bias,
    const torch::Tensor& iou_proj_out_weight,
    const torch::Tensor& iou_proj_out_bias,
    const torch::Tensor& iou_hidden1,
    const torch::Tensor& iou_hidden1_relu,
    const torch::Tensor& iou_hidden2,
    const torch::Tensor& iou_hidden2_relu,
    const torch::Tensor& hyper_proj_in_weights,
    const torch::Tensor& hyper_proj_in_biases,
    const torch::Tensor& hyper_hidden_weights,
    const torch::Tensor& hyper_hidden_biases,
    const torch::Tensor& hyper_proj_out_weights,
    const torch::Tensor& hyper_proj_out_biases,
    const torch::Tensor& hyper_hidden1_0,
    const torch::Tensor& hyper_hidden1_1,
    const torch::Tensor& hyper_hidden1_2,
    const torch::Tensor& hyper_hidden1_3,
    const torch::Tensor& hyper_hidden1_relu_0,
    const torch::Tensor& hyper_hidden1_relu_1,
    const torch::Tensor& hyper_hidden1_relu_2,
    const torch::Tensor& hyper_hidden1_relu_3,
    const torch::Tensor& hyper_hidden2_0,
    const torch::Tensor& hyper_hidden2_1,
    const torch::Tensor& hyper_hidden2_2,
    const torch::Tensor& hyper_hidden2_3,
    const torch::Tensor& hyper_hidden2_relu_0,
    const torch::Tensor& hyper_hidden2_relu_1,
    const torch::Tensor& hyper_hidden2_relu_2,
    const torch::Tensor& hyper_hidden2_relu_3,
    torch::Tensor grad_iou_token_out,
    torch::Tensor grad_mask_tokens_out,
    torch::Tensor grad_iou_proj_in_weight,
    torch::Tensor grad_iou_proj_in_bias,
    torch::Tensor grad_iou_hidden_weight,
    torch::Tensor grad_iou_hidden_bias,
    torch::Tensor grad_iou_proj_out_weight,
    torch::Tensor grad_iou_proj_out_bias,
    torch::Tensor grad_hyper_proj_in_weights,
    torch::Tensor grad_hyper_proj_in_biases,
    torch::Tensor grad_hyper_hidden_weights,
    torch::Tensor grad_hyper_hidden_biases,
    torch::Tensor grad_hyper_proj_out_weights,
    torch::Tensor grad_hyper_proj_out_biases
) {
    check_fp32_cuda(grad_iou_scores, "grad_iou_scores");
    check_fp32_cuda(grad_hyper_weights, "grad_hyper_weights");
    check_fp32_cuda(iou_token_out, "iou_token_out");
    check_fp32_cuda(mask_tokens_out, "mask_tokens_out");
    check_fp32_cuda(iou_proj_in_weight, "iou_proj_in_weight");
    check_fp32_cuda(iou_hidden_weight, "iou_hidden_weight");
    check_fp32_cuda(iou_proj_out_weight, "iou_proj_out_weight");
    check_fp32_cuda(iou_hidden1, "iou_hidden1");
    check_fp32_cuda(iou_hidden1_relu, "iou_hidden1_relu");
    check_fp32_cuda(iou_hidden2, "iou_hidden2");
    check_fp32_cuda(iou_hidden2_relu, "iou_hidden2_relu");
    check_fp32_cuda(hyper_proj_in_weights, "hyper_proj_in_weights");
    check_fp32_cuda(hyper_hidden_weights, "hyper_hidden_weights");
    check_fp32_cuda(hyper_proj_out_weights, "hyper_proj_out_weights");

    TORCH_CHECK(iou_token_out.dim() == 3,
                "iou_token_out must be 3D [B, P, hidden_size], got ", iou_token_out.dim(), "D");
    TORCH_CHECK(mask_tokens_out.dim() == 4,
                "mask_tokens_out must be 4D [B, P, num_mask_tokens, hidden_size], got ",
                mask_tokens_out.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_iou_scores.device());

    const int64_t batch_size = iou_token_out.size(0);
    const int64_t point_batch_size = iou_token_out.size(1);
    const int64_t hidden_size = iou_token_out.size(2);
    const int64_t iou_head_hidden_dim = iou_hidden1.size(2);
    const int64_t num_mask_tokens = mask_tokens_out.size(2);
    const int64_t output_channels = grad_hyper_weights.size(3);

    TORCH_CHECK(num_mask_tokens == 4,
                "this port assumes num_mask_tokens == 4, got ", num_mask_tokens);

    // ==================== IoU path backward ====================
    // Gradient through IoU output layer: Linear(1024 -> 4).
    auto grad_iou_scores_flat = grad_iou_scores.reshape({-1, num_mask_tokens});
    auto iou_hidden2_relu_flat = iou_hidden2_relu.reshape({-1, iou_head_hidden_dim});

    auto grad_iou_proj_out_weight_result = grad_iou_scores_flat.t().mm(iou_hidden2_relu_flat);
    auto grad_iou_proj_out_bias_result = grad_iou_scores_flat.sum(/*dim=*/0);

    // Gradient w.r.t. iou_hidden2_relu, then through ReLU via the pre-ReLU mask.
    auto grad_iou_hidden2_relu = grad_iou_scores_flat.mm(iou_proj_out_weight)
                                     .reshape({batch_size, point_batch_size, iou_head_hidden_dim});
    auto grad_iou_hidden2 = grad_iou_hidden2_relu * (iou_hidden2 > 0).to(torch::kFloat32);

    // Gradient through IoU hidden layer: Linear(1024 -> 1024).
    auto grad_iou_hidden2_flat = grad_iou_hidden2.reshape({-1, iou_head_hidden_dim});
    auto iou_hidden1_relu_flat = iou_hidden1_relu.reshape({-1, iou_head_hidden_dim});

    auto grad_iou_hidden_weight_result = grad_iou_hidden2_flat.t().mm(iou_hidden1_relu_flat);
    auto grad_iou_hidden_bias_result = grad_iou_hidden2_flat.sum(/*dim=*/0);

    auto grad_iou_hidden1_relu = grad_iou_hidden2_flat.mm(iou_hidden_weight)
                                     .reshape({batch_size, point_batch_size, iou_head_hidden_dim});
    auto grad_iou_hidden1 = grad_iou_hidden1_relu * (iou_hidden1 > 0).to(torch::kFloat32);

    // Gradient through IoU input layer: Linear(256 -> 1024).
    auto grad_iou_hidden1_flat = grad_iou_hidden1.reshape({-1, iou_head_hidden_dim});
    auto iou_token_out_flat = iou_token_out.reshape({-1, hidden_size});

    auto grad_iou_proj_in_weight_result = grad_iou_hidden1_flat.t().mm(iou_token_out_flat);
    auto grad_iou_proj_in_bias_result = grad_iou_hidden1_flat.sum(/*dim=*/0);

    auto grad_iou_token_out_result = grad_iou_hidden1_flat.mm(iou_proj_in_weight)
                                         .reshape({batch_size, point_batch_size, hidden_size});

    // ==================== Hypernetwork path backward ====================
    auto grad_mask_tokens_out_result = at::zeros_like(mask_tokens_out);
    auto grad_hyper_proj_in_weights_result = at::zeros_like(hyper_proj_in_weights);
    auto grad_hyper_proj_in_biases_result = at::zeros_like(hyper_proj_in_biases);
    auto grad_hyper_hidden_weights_result = at::zeros_like(hyper_hidden_weights);
    auto grad_hyper_hidden_biases_result = at::zeros_like(hyper_hidden_biases);
    auto grad_hyper_proj_out_weights_result = at::zeros_like(hyper_proj_out_weights);
    auto grad_hyper_proj_out_biases_result = at::zeros_like(hyper_proj_out_biases);

    const std::array<const torch::Tensor*, 4> hyper_hidden1_list = {
        &hyper_hidden1_0, &hyper_hidden1_1, &hyper_hidden1_2, &hyper_hidden1_3};
    const std::array<const torch::Tensor*, 4> hyper_hidden1_relu_list = {
        &hyper_hidden1_relu_0, &hyper_hidden1_relu_1, &hyper_hidden1_relu_2, &hyper_hidden1_relu_3};
    const std::array<const torch::Tensor*, 4> hyper_hidden2_list = {
        &hyper_hidden2_0, &hyper_hidden2_1, &hyper_hidden2_2, &hyper_hidden2_3};
    const std::array<const torch::Tensor*, 4> hyper_hidden2_relu_list = {
        &hyper_hidden2_relu_0, &hyper_hidden2_relu_1, &hyper_hidden2_relu_2, &hyper_hidden2_relu_3};

    for (int64_t mask_idx = 0; mask_idx < num_mask_tokens; ++mask_idx) {
        // Extract upstream gradient and saved activations for this mask.
        auto grad_weights = at::select(grad_hyper_weights, /*dim=*/2, mask_idx);
        const auto& hidden1 = *hyper_hidden1_list[mask_idx];
        const auto& hidden1_relu = *hyper_hidden1_relu_list[mask_idx];
        const auto& hidden2 = *hyper_hidden2_list[mask_idx];
        const auto& hidden2_relu = *hyper_hidden2_relu_list[mask_idx];
        auto token = at::select(mask_tokens_out, /*dim=*/2, mask_idx);

        auto grad_weights_flat = grad_weights.reshape({-1, output_channels});
        auto hidden2_relu_flat = hidden2_relu.reshape({-1, hidden_size});
        auto hidden1_relu_flat = hidden1_relu.reshape({-1, hidden_size});
        auto token_flat = token.reshape({-1, hidden_size});

        // Gradient through output layer: Linear(256 -> 32).
        auto grad_proj_out_weight = grad_weights_flat.t().mm(hidden2_relu_flat);
        auto grad_proj_out_bias = grad_weights_flat.sum(/*dim=*/0);
        grad_hyper_proj_out_weights_result.select(/*dim=*/0, mask_idx).copy_(grad_proj_out_weight);
        grad_hyper_proj_out_biases_result.select(/*dim=*/0, mask_idx).copy_(grad_proj_out_bias);

        // Gradient w.r.t. hidden2_relu, then through ReLU.
        auto grad_hidden2_relu = grad_weights_flat.mm(at::select(hyper_proj_out_weights, /*dim=*/0, mask_idx))
                                     .reshape({batch_size, point_batch_size, hidden_size});
        auto grad_hidden2 = grad_hidden2_relu * (hidden2 > 0).to(torch::kFloat32);
        auto grad_hidden2_flat = grad_hidden2.reshape({-1, hidden_size});

        // Gradient through hidden layer: Linear(256 -> 256).
        auto grad_hidden_weight = grad_hidden2_flat.t().mm(hidden1_relu_flat);
        auto grad_hidden_bias = grad_hidden2_flat.sum(/*dim=*/0);
        grad_hyper_hidden_weights_result.select(/*dim=*/0, mask_idx).copy_(grad_hidden_weight);
        grad_hyper_hidden_biases_result.select(/*dim=*/0, mask_idx).copy_(grad_hidden_bias);

        // Gradient w.r.t. hidden1_relu, then through ReLU.
        auto grad_hidden1_relu = grad_hidden2_flat.mm(at::select(hyper_hidden_weights, /*dim=*/0, mask_idx))
                                     .reshape({batch_size, point_batch_size, hidden_size});
        auto grad_hidden1 = grad_hidden1_relu * (hidden1 > 0).to(torch::kFloat32);
        auto grad_hidden1_flat = grad_hidden1.reshape({-1, hidden_size});

        // Gradient through input layer: Linear(256 -> 256).
        auto grad_proj_in_weight = grad_hidden1_flat.t().mm(token_flat);
        auto grad_proj_in_bias = grad_hidden1_flat.sum(/*dim=*/0);
        grad_hyper_proj_in_weights_result.select(/*dim=*/0, mask_idx).copy_(grad_proj_in_weight);
        grad_hyper_proj_in_biases_result.select(/*dim=*/0, mask_idx).copy_(grad_proj_in_bias);

        // Gradient w.r.t. the mask token input.
        auto grad_token = grad_hidden1_flat.mm(at::select(hyper_proj_in_weights, /*dim=*/0, mask_idx))
                              .reshape({batch_size, point_batch_size, hidden_size});
        grad_mask_tokens_out_result.select(/*dim=*/2, mask_idx).copy_(grad_token);
    }

    // Destination-passing-style writes (definition.json output order).
    TORCH_CHECK(grad_iou_token_out_result.sizes() == grad_iou_token_out.sizes(),
                "grad_iou_token_out shape ", grad_iou_token_out_result.sizes(),
                " does not match pre-allocated ", grad_iou_token_out.sizes());
    TORCH_CHECK(grad_mask_tokens_out_result.sizes() == grad_mask_tokens_out.sizes(),
                "grad_mask_tokens_out shape ", grad_mask_tokens_out_result.sizes(),
                " does not match pre-allocated ", grad_mask_tokens_out.sizes());
    TORCH_CHECK(grad_iou_proj_in_weight_result.sizes() == grad_iou_proj_in_weight.sizes(),
                "grad_iou_proj_in_weight shape ", grad_iou_proj_in_weight_result.sizes(),
                " does not match pre-allocated ", grad_iou_proj_in_weight.sizes());
    TORCH_CHECK(grad_iou_proj_in_bias_result.sizes() == grad_iou_proj_in_bias.sizes(),
                "grad_iou_proj_in_bias shape ", grad_iou_proj_in_bias_result.sizes(),
                " does not match pre-allocated ", grad_iou_proj_in_bias.sizes());
    TORCH_CHECK(grad_iou_hidden_weight_result.sizes() == grad_iou_hidden_weight.sizes(),
                "grad_iou_hidden_weight shape ", grad_iou_hidden_weight_result.sizes(),
                " does not match pre-allocated ", grad_iou_hidden_weight.sizes());
    TORCH_CHECK(grad_iou_hidden_bias_result.sizes() == grad_iou_hidden_bias.sizes(),
                "grad_iou_hidden_bias shape ", grad_iou_hidden_bias_result.sizes(),
                " does not match pre-allocated ", grad_iou_hidden_bias.sizes());
    TORCH_CHECK(grad_iou_proj_out_weight_result.sizes() == grad_iou_proj_out_weight.sizes(),
                "grad_iou_proj_out_weight shape ", grad_iou_proj_out_weight_result.sizes(),
                " does not match pre-allocated ", grad_iou_proj_out_weight.sizes());
    TORCH_CHECK(grad_iou_proj_out_bias_result.sizes() == grad_iou_proj_out_bias.sizes(),
                "grad_iou_proj_out_bias shape ", grad_iou_proj_out_bias_result.sizes(),
                " does not match pre-allocated ", grad_iou_proj_out_bias.sizes());
    TORCH_CHECK(grad_hyper_proj_in_weights_result.sizes() == grad_hyper_proj_in_weights.sizes(),
                "grad_hyper_proj_in_weights shape ", grad_hyper_proj_in_weights_result.sizes(),
                " does not match pre-allocated ", grad_hyper_proj_in_weights.sizes());
    TORCH_CHECK(grad_hyper_proj_in_biases_result.sizes() == grad_hyper_proj_in_biases.sizes(),
                "grad_hyper_proj_in_biases shape ", grad_hyper_proj_in_biases_result.sizes(),
                " does not match pre-allocated ", grad_hyper_proj_in_biases.sizes());
    TORCH_CHECK(grad_hyper_hidden_weights_result.sizes() == grad_hyper_hidden_weights.sizes(),
                "grad_hyper_hidden_weights shape ", grad_hyper_hidden_weights_result.sizes(),
                " does not match pre-allocated ", grad_hyper_hidden_weights.sizes());
    TORCH_CHECK(grad_hyper_hidden_biases_result.sizes() == grad_hyper_hidden_biases.sizes(),
                "grad_hyper_hidden_biases shape ", grad_hyper_hidden_biases_result.sizes(),
                " does not match pre-allocated ", grad_hyper_hidden_biases.sizes());
    TORCH_CHECK(grad_hyper_proj_out_weights_result.sizes() == grad_hyper_proj_out_weights.sizes(),
                "grad_hyper_proj_out_weights shape ", grad_hyper_proj_out_weights_result.sizes(),
                " does not match pre-allocated ", grad_hyper_proj_out_weights.sizes());
    TORCH_CHECK(grad_hyper_proj_out_biases_result.sizes() == grad_hyper_proj_out_biases.sizes(),
                "grad_hyper_proj_out_biases shape ", grad_hyper_proj_out_biases_result.sizes(),
                " does not match pre-allocated ", grad_hyper_proj_out_biases.sizes());

    grad_iou_token_out.copy_(grad_iou_token_out_result);
    grad_mask_tokens_out.copy_(grad_mask_tokens_out_result);
    grad_iou_proj_in_weight.copy_(grad_iou_proj_in_weight_result);
    grad_iou_proj_in_bias.copy_(grad_iou_proj_in_bias_result);
    grad_iou_hidden_weight.copy_(grad_iou_hidden_weight_result);
    grad_iou_hidden_bias.copy_(grad_iou_hidden_bias_result);
    grad_iou_proj_out_weight.copy_(grad_iou_proj_out_weight_result);
    grad_iou_proj_out_bias.copy_(grad_iou_proj_out_bias_result);
    grad_hyper_proj_in_weights.copy_(grad_hyper_proj_in_weights_result);
    grad_hyper_proj_in_biases.copy_(grad_hyper_proj_in_biases_result);
    grad_hyper_hidden_weights.copy_(grad_hyper_hidden_weights_result);
    grad_hyper_hidden_biases.copy_(grad_hyper_hidden_biases_result);
    grad_hyper_proj_out_weights.copy_(grad_hyper_proj_out_weights_result);
    grad_hyper_proj_out_biases.copy_(grad_hyper_proj_out_biases_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Backward pass for SAM-HQ mask decoder IoU prediction and "
        "hypernetwork weight generation. Pure ATen / cuBLAS dispatch: the "
        "analytic grad algebra for the IoU 3-layer ReLU MLP plus the four "
        "parallel hypernetwork 3-layer ReLU MLPs reproduced op-for-op from "
        "the reference (linear backward grad_W = grad_out.T @ in, grad_b = "
        "grad_out.sum(0), grad_in = grad_out @ W; ReLU grad masked on the "
        "saved pre-ReLU activations), native float32 throughout, four experts "
        "in a host loop. Destination-passing style (14 outputs).");
}
