#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <vector>

// Fused IoU prediction and hypernetwork weight generation for the SAM-HQ mask
// decoder.
//
// Reference (problem_086_sam_hq_mask_decoder_iou_hypernetwork_fusion),
// reproduced op-for-op:
//
//   IoU prediction path -- a 3-layer perceptron with ReLU between the two
//   leading linear projections, consuming the IoU token embedding:
//     iou_h1     = relu( linear(iou_token_out, iou_proj_in_weight,  iou_proj_in_bias) )   # 256 -> 1024
//     iou_h2     = relu( linear(iou_h1,        iou_hidden_weight,   iou_hidden_bias)  )   # 1024 -> 1024
//     iou_scores =       linear(iou_h2,        iou_proj_out_weight, iou_proj_out_bias)    # 1024 -> num_mask_tokens (4)
//   The final projection has no activation.
//
//   Hypernetwork path -- four independent 3-layer perceptrons, one per mask
//   token. Hypernetwork i consumes the i-th mask-token slice
//   mask_tokens_out[:, :, i, :] and produces a per-token weight vector:
//     token_i = mask_tokens_out[:, :, i, :]                                            # [B, P, hidden_size]
//     h_i_1   = relu( linear(token_i, hyperI_proj_in_weight,  hyperI_proj_in_bias)  )  # 256 -> 256
//     h_i_2   = relu( linear(h_i_1,   hyperI_hidden_weight,   hyperI_hidden_bias)   )  # 256 -> 256
//     w_i     =       linear(h_i_2,   hyperI_proj_out_weight, hyperI_proj_out_bias)    # 256 -> output_channels (32)
//   The four per-token weight vectors w0..w3 are stacked along a new axis
//   inserted at position 2, giving hyper_weights of shape
//   [B, P, num_mask_tokens, output_channels].
//
// Per definition.json (insertion order):
//   iou_token_out        : [batch_size, point_batch_size, hidden_size=256]                 fp32
//   mask_tokens_out      : [batch_size, point_batch_size, num_mask_tokens=4, hidden_size]  fp32
//   iou_proj_in_weight   : [iou_head_hidden_dim=1024, hidden_size=256]                     fp32
//   iou_proj_in_bias     : [iou_head_hidden_dim=1024]                                      fp32
//   iou_hidden_weight    : [iou_head_hidden_dim, iou_head_hidden_dim]                      fp32
//   iou_hidden_bias      : [iou_head_hidden_dim]                                           fp32
//   iou_proj_out_weight  : [num_mask_tokens=4, iou_head_hidden_dim]                        fp32
//   iou_proj_out_bias    : [num_mask_tokens=4]                                             fp32
//   hyperN_proj_in_weight   : [hidden_size, hidden_size]      (N in 0..3)                  fp32
//   hyperN_proj_in_bias     : [hidden_size]                                                fp32
//   hyperN_hidden_weight    : [hidden_size, hidden_size]                                   fp32
//   hyperN_hidden_bias      : [hidden_size]                                                fp32
//   hyperN_proj_out_weight  : [output_channels=32, hidden_size]                            fp32
//   hyperN_proj_out_bias    : [output_channels=32]                                         fp32
//   iou_scores    : [batch_size, point_batch_size, num_mask_tokens=4]                      fp32 DPS
//   hyper_weights : [batch_size, point_batch_size, num_mask_tokens=4, output_channels=32]  fp32 DPS
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. Each
// linear layer is at::linear (weight already in [out, in] layout, applied as
// x @ weight.t() + bias), each activation is at::relu, the per-token mask
// slices come from at::select on axis 2, and the four hypernetwork outputs are
// combined with at::stack along axis 2. Everything stays in the native fp32
// dtype, so there are no cast points to track. Both outputs are written in
// place into their pre-allocated destinations (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& iou_token_out,
    const torch::Tensor& mask_tokens_out,
    const torch::Tensor& iou_proj_in_weight,
    const torch::Tensor& iou_proj_in_bias,
    const torch::Tensor& iou_hidden_weight,
    const torch::Tensor& iou_hidden_bias,
    const torch::Tensor& iou_proj_out_weight,
    const torch::Tensor& iou_proj_out_bias,
    const torch::Tensor& hyper0_proj_in_weight,
    const torch::Tensor& hyper0_proj_in_bias,
    const torch::Tensor& hyper0_hidden_weight,
    const torch::Tensor& hyper0_hidden_bias,
    const torch::Tensor& hyper0_proj_out_weight,
    const torch::Tensor& hyper0_proj_out_bias,
    const torch::Tensor& hyper1_proj_in_weight,
    const torch::Tensor& hyper1_proj_in_bias,
    const torch::Tensor& hyper1_hidden_weight,
    const torch::Tensor& hyper1_hidden_bias,
    const torch::Tensor& hyper1_proj_out_weight,
    const torch::Tensor& hyper1_proj_out_bias,
    const torch::Tensor& hyper2_proj_in_weight,
    const torch::Tensor& hyper2_proj_in_bias,
    const torch::Tensor& hyper2_hidden_weight,
    const torch::Tensor& hyper2_hidden_bias,
    const torch::Tensor& hyper2_proj_out_weight,
    const torch::Tensor& hyper2_proj_out_bias,
    const torch::Tensor& hyper3_proj_in_weight,
    const torch::Tensor& hyper3_proj_in_bias,
    const torch::Tensor& hyper3_hidden_weight,
    const torch::Tensor& hyper3_hidden_bias,
    const torch::Tensor& hyper3_proj_out_weight,
    const torch::Tensor& hyper3_proj_out_bias,
    torch::Tensor iou_scores,
    torch::Tensor hyper_weights
) {
    check_fp32_cuda(iou_token_out, "iou_token_out");
    check_fp32_cuda(mask_tokens_out, "mask_tokens_out");
    check_fp32_cuda(iou_proj_in_weight, "iou_proj_in_weight");
    check_fp32_cuda(iou_proj_in_bias, "iou_proj_in_bias");
    check_fp32_cuda(iou_hidden_weight, "iou_hidden_weight");
    check_fp32_cuda(iou_hidden_bias, "iou_hidden_bias");
    check_fp32_cuda(iou_proj_out_weight, "iou_proj_out_weight");
    check_fp32_cuda(iou_proj_out_bias, "iou_proj_out_bias");
    check_fp32_cuda(hyper0_proj_in_weight, "hyper0_proj_in_weight");
    check_fp32_cuda(hyper0_proj_in_bias, "hyper0_proj_in_bias");
    check_fp32_cuda(hyper0_hidden_weight, "hyper0_hidden_weight");
    check_fp32_cuda(hyper0_hidden_bias, "hyper0_hidden_bias");
    check_fp32_cuda(hyper0_proj_out_weight, "hyper0_proj_out_weight");
    check_fp32_cuda(hyper0_proj_out_bias, "hyper0_proj_out_bias");
    check_fp32_cuda(hyper1_proj_in_weight, "hyper1_proj_in_weight");
    check_fp32_cuda(hyper1_proj_in_bias, "hyper1_proj_in_bias");
    check_fp32_cuda(hyper1_hidden_weight, "hyper1_hidden_weight");
    check_fp32_cuda(hyper1_hidden_bias, "hyper1_hidden_bias");
    check_fp32_cuda(hyper1_proj_out_weight, "hyper1_proj_out_weight");
    check_fp32_cuda(hyper1_proj_out_bias, "hyper1_proj_out_bias");
    check_fp32_cuda(hyper2_proj_in_weight, "hyper2_proj_in_weight");
    check_fp32_cuda(hyper2_proj_in_bias, "hyper2_proj_in_bias");
    check_fp32_cuda(hyper2_hidden_weight, "hyper2_hidden_weight");
    check_fp32_cuda(hyper2_hidden_bias, "hyper2_hidden_bias");
    check_fp32_cuda(hyper2_proj_out_weight, "hyper2_proj_out_weight");
    check_fp32_cuda(hyper2_proj_out_bias, "hyper2_proj_out_bias");
    check_fp32_cuda(hyper3_proj_in_weight, "hyper3_proj_in_weight");
    check_fp32_cuda(hyper3_proj_in_bias, "hyper3_proj_in_bias");
    check_fp32_cuda(hyper3_hidden_weight, "hyper3_hidden_weight");
    check_fp32_cuda(hyper3_hidden_bias, "hyper3_hidden_bias");
    check_fp32_cuda(hyper3_proj_out_weight, "hyper3_proj_out_weight");
    check_fp32_cuda(hyper3_proj_out_bias, "hyper3_proj_out_bias");
    check_fp32_cuda(iou_scores, "iou_scores");
    check_fp32_cuda(hyper_weights, "hyper_weights");

    TORCH_CHECK(iou_token_out.dim() == 3,
                "iou_token_out must be 3D [B, P, hidden_size], got ", iou_token_out.dim(), "D");
    TORCH_CHECK(mask_tokens_out.dim() == 4,
                "mask_tokens_out must be 4D [B, P, num_mask_tokens, hidden_size], got ",
                mask_tokens_out.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(iou_token_out.device());

    const int64_t num_mask_tokens = mask_tokens_out.size(2);
    TORCH_CHECK(num_mask_tokens == 4,
                "this port assumes num_mask_tokens == 4, got ", num_mask_tokens);

    // IoU prediction path: two ReLU-activated projections followed by an
    // unactivated output projection mapping the IoU token to per-token scores.
    auto iou_h1 = at::relu(at::linear(iou_token_out, iou_proj_in_weight, iou_proj_in_bias));
    auto iou_h2 = at::relu(at::linear(iou_h1, iou_hidden_weight, iou_hidden_bias));
    auto iou_scores_result = at::linear(iou_h2, iou_proj_out_weight, iou_proj_out_bias);

    // Hypernetwork path: four independent 3-layer perceptrons, each fed the
    // matching mask-token slice along axis 2. The leading two projections use
    // ReLU; the output projection is unactivated.
    auto token0 = at::select(mask_tokens_out, /*dim=*/2, /*index=*/0);
    auto h0_1 = at::relu(at::linear(token0, hyper0_proj_in_weight, hyper0_proj_in_bias));
    auto h0_2 = at::relu(at::linear(h0_1, hyper0_hidden_weight, hyper0_hidden_bias));
    auto w0 = at::linear(h0_2, hyper0_proj_out_weight, hyper0_proj_out_bias);

    auto token1 = at::select(mask_tokens_out, /*dim=*/2, /*index=*/1);
    auto h1_1 = at::relu(at::linear(token1, hyper1_proj_in_weight, hyper1_proj_in_bias));
    auto h1_2 = at::relu(at::linear(h1_1, hyper1_hidden_weight, hyper1_hidden_bias));
    auto w1 = at::linear(h1_2, hyper1_proj_out_weight, hyper1_proj_out_bias);

    auto token2 = at::select(mask_tokens_out, /*dim=*/2, /*index=*/2);
    auto h2_1 = at::relu(at::linear(token2, hyper2_proj_in_weight, hyper2_proj_in_bias));
    auto h2_2 = at::relu(at::linear(h2_1, hyper2_hidden_weight, hyper2_hidden_bias));
    auto w2 = at::linear(h2_2, hyper2_proj_out_weight, hyper2_proj_out_bias);

    auto token3 = at::select(mask_tokens_out, /*dim=*/2, /*index=*/3);
    auto h3_1 = at::relu(at::linear(token3, hyper3_proj_in_weight, hyper3_proj_in_bias));
    auto h3_2 = at::relu(at::linear(h3_1, hyper3_hidden_weight, hyper3_hidden_bias));
    auto w3 = at::linear(h3_2, hyper3_proj_out_weight, hyper3_proj_out_bias);

    // Combine the four per-token weight vectors along a new axis at position 2,
    // matching the reference's torch.stack(dim=2).
    std::vector<torch::Tensor> hyper_list = {w0, w1, w2, w3};
    auto hyper_weights_result = at::stack(hyper_list, /*dim=*/2);

    // Destination-passing-style writes for both outputs.
    TORCH_CHECK(iou_scores_result.sizes() == iou_scores.sizes(),
                "iou_scores shape ", iou_scores_result.sizes(),
                " does not match pre-allocated ", iou_scores.sizes());
    TORCH_CHECK(hyper_weights_result.sizes() == hyper_weights.sizes(),
                "hyper_weights shape ", hyper_weights_result.sizes(),
                " does not match pre-allocated ", hyper_weights.sizes());
    iou_scores.copy_(iou_scores_result);
    hyper_weights.copy_(hyper_weights_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "SAM-HQ fused IoU prediction and hypernetwork weight generation via "
          "pure ATen dispatch: a 3-layer ReLU MLP for IoU scores plus four "
          "parallel 3-layer ReLU MLPs over the mask-token slices, stacked on "
          "axis 2. fp32; destination-passing style.");
}
