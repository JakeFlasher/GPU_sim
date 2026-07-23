#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// Edit consistency loss combining a spatially-weighted pixel reconstruction
// term, a multi-scale perceptual-feature term, and a preservation term for the
// non-edited region. Reproduced op-for-op from the reference run().
//
// Spatial weighting comes from the edit mask plus a Sobel edge map of that
// mask:
//
//   sobel_x          = [[-1,0,1],[-2,0,2],[-1,0,1]]            (1,1,3,3)
//   sobel_y          = [[-1,-2,-1],[0,0,0],[1,2,1]]            (1,1,3,3)
//   edges_x          = conv2d(edit_mask, sobel_x, padding=1)   (no bias)
//   edges_y          = conv2d(edit_mask, sobel_y, padding=1)   (no bias)
//   edge_magnitude   = sqrt(edges_x^2 + edges_y^2 + 1e-8)
//   spatial_weights  = 1 + (edit_region_weight_multiplier - 1)*edit_mask
//                        + 0.5 * edge_magnitude
//
// Pixel term (L1, spatially weighted):
//
//   pixel_loss = mean( |predicted - target| * spatial_weights )
//
// Perceptual term: three feature extractors run as a cascade. Extractor 0
// consumes the raw images; extractor 1 consumes extractor 0's pooled output;
// extractor 2 consumes extractor 1's pooled output. Each extractor is
//
//   conv2d(.,conv1_w,conv1_b,padding=1) -> relu
//     -> conv2d(.,conv2_w,conv2_b,padding=1) -> relu
//     -> max_pool2d(kernel=2, stride=2)
//
// and is applied to both predicted and target with the same weights. For each
// of the three scales the spatial weights are bilinearly resized (align_corners
// = false) to the feature resolution, the weighted L1 feature difference is
// averaged, scaled by |scale_weights[i]|, and summed:
//
//   downsampled = interpolate(spatial_weights, (Hf,Wf), bilinear, align=false)
//   scale_loss  = mean( |pred_feat - target_feat| * downsampled )
//   perceptual_loss += scale_loss * |scale_weights[i]|
//
// Preservation term over the non-edited region:
//
//   preservation_loss = mean( |predicted - source| * (1 - edit_mask) )
//
// Combination with the three scalar weights (the 0.3 preservation coefficient
// is fixed in the reference):
//
//   total_loss = pixel_loss_weight       * pixel_loss
//              + perceptual_loss_weight   * perceptual_loss
//              + 0.3                      * preservation_loss
//
// Per definition.json (insertion order):
//   predicted               : [batch_size, 3, height, width]   fp32
//   target                  : [batch_size, 3, height, width]   fp32
//   source                  : [batch_size, 3, height, width]   fp32
//   edit_mask               : [batch_size, 1, height, width]   fp32
//   extractor0_conv1_weight : [64, 3, 3, 3]                    fp32
//   extractor0_conv1_bias   : [64]                             fp32
//   extractor0_conv2_weight : [64, 64, 3, 3]                   fp32
//   extractor0_conv2_bias   : [64]                             fp32
//   extractor1_conv1_weight : [128, 64, 3, 3]                  fp32
//   extractor1_conv1_bias   : [128]                            fp32
//   extractor1_conv2_weight : [128, 128, 3, 3]                 fp32
//   extractor1_conv2_bias   : [128]                            fp32
//   extractor2_conv1_weight : [256, 128, 3, 3]                 fp32
//   extractor2_conv1_bias   : [256]                            fp32
//   extractor2_conv2_weight : [256, 256, 3, 3]                 fp32
//   extractor2_conv2_bias   : [256]                            fp32
//   scale_weights           : [3]                              fp32
//   pixel_loss_weight             : scalar                     fp32 (passed as double)
//   perceptual_loss_weight        : scalar                     fp32 (passed as double)
//   edit_region_weight_multiplier : scalar                     fp32 (passed as double)
//   total_loss              : []                               fp32 DPS (pre-alloc)
//
// Implementation: pure ATen / cuDNN dispatch -- NO custom CUDA kernels. The
// convolutions, pooling, bilinear resize, and reductions all route through
// ATen at fp32, matching the reference's precision profile. The scalar result
// is written in place into the pre-allocated 0-dim destination
// (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

// One feature extractor: two padded 3x3 convolutions each followed by relu,
// then a 2x2 max-pool. Applied to predicted and target with shared weights.
static torch::Tensor extract_features(
    const torch::Tensor& x,
    const torch::Tensor& conv1_weight,
    const torch::Tensor& conv1_bias,
    const torch::Tensor& conv2_weight,
    const torch::Tensor& conv2_bias,
    const std::vector<int64_t>& stride1,
    const std::vector<int64_t>& pad1,
    const std::vector<int64_t>& pool_kernel,
    const std::vector<int64_t>& pool_stride
) {
    auto y = at::conv2d(x, conv1_weight, conv1_bias, /*stride=*/stride1, /*padding=*/pad1);
    y = at::relu(y);
    y = at::conv2d(y, conv2_weight, conv2_bias, /*stride=*/stride1, /*padding=*/pad1);
    y = at::relu(y);
    y = at::max_pool2d(y, /*kernel_size=*/pool_kernel, /*stride=*/pool_stride);
    return y;
}

void run(
    const torch::Tensor& predicted,
    const torch::Tensor& target,
    const torch::Tensor& source,
    const torch::Tensor& edit_mask,
    const torch::Tensor& extractor0_conv1_weight,
    const torch::Tensor& extractor0_conv1_bias,
    const torch::Tensor& extractor0_conv2_weight,
    const torch::Tensor& extractor0_conv2_bias,
    const torch::Tensor& extractor1_conv1_weight,
    const torch::Tensor& extractor1_conv1_bias,
    const torch::Tensor& extractor1_conv2_weight,
    const torch::Tensor& extractor1_conv2_bias,
    const torch::Tensor& extractor2_conv1_weight,
    const torch::Tensor& extractor2_conv1_bias,
    const torch::Tensor& extractor2_conv2_weight,
    const torch::Tensor& extractor2_conv2_bias,
    const torch::Tensor& scale_weights,
    double pixel_loss_weight,
    double perceptual_loss_weight,
    double edit_region_weight_multiplier,
    torch::Tensor total_loss
) {
    check_fp32_cuda(predicted, "predicted");
    check_fp32_cuda(target, "target");
    check_fp32_cuda(source, "source");
    check_fp32_cuda(edit_mask, "edit_mask");
    check_fp32_cuda(extractor0_conv1_weight, "extractor0_conv1_weight");
    check_fp32_cuda(extractor0_conv1_bias, "extractor0_conv1_bias");
    check_fp32_cuda(extractor0_conv2_weight, "extractor0_conv2_weight");
    check_fp32_cuda(extractor0_conv2_bias, "extractor0_conv2_bias");
    check_fp32_cuda(extractor1_conv1_weight, "extractor1_conv1_weight");
    check_fp32_cuda(extractor1_conv1_bias, "extractor1_conv1_bias");
    check_fp32_cuda(extractor1_conv2_weight, "extractor1_conv2_weight");
    check_fp32_cuda(extractor1_conv2_bias, "extractor1_conv2_bias");
    check_fp32_cuda(extractor2_conv1_weight, "extractor2_conv1_weight");
    check_fp32_cuda(extractor2_conv1_bias, "extractor2_conv1_bias");
    check_fp32_cuda(extractor2_conv2_weight, "extractor2_conv2_weight");
    check_fp32_cuda(extractor2_conv2_bias, "extractor2_conv2_bias");
    check_fp32_cuda(scale_weights, "scale_weights");
    check_fp32_cuda(total_loss, "total_loss");

    TORCH_CHECK(predicted.dim() == 4, "predicted must be 4D [N, C, H, W], got ", predicted.dim(), "D");
    TORCH_CHECK(edit_mask.dim() == 4, "edit_mask must be 4D [N, 1, H, W], got ", edit_mask.dim(), "D");
    TORCH_CHECK(scale_weights.numel() >= 3,
                "scale_weights must hold at least 3 entries, got ", scale_weights.numel());

    const c10::cuda::CUDAGuard device_guard(predicted.device());

    const auto opts = edit_mask.options();

    const std::vector<int64_t> stride1{1, 1};
    const std::vector<int64_t> pad1{1, 1};
    const std::vector<int64_t> pool_kernel{2, 2};
    const std::vector<int64_t> pool_stride{2, 2};

    // Sobel edge filters built as constant 1x1x3x3 kernels on the mask's device
    // and dtype; the edge convolutions carry no bias.
    auto sobel_x = torch::tensor(
        {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f}, opts).view({1, 1, 3, 3});
    auto sobel_y = torch::tensor(
        {-1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f}, opts).view({1, 1, 3, 3});

    auto edges_x = at::conv2d(edit_mask, sobel_x, /*bias=*/{}, /*stride=*/stride1, /*padding=*/pad1);
    auto edges_y = at::conv2d(edit_mask, sobel_y, /*bias=*/{}, /*stride=*/stride1, /*padding=*/pad1);
    auto edge_magnitude = at::sqrt(edges_x * edges_x + edges_y * edges_y + 1e-8);

    // Spatial weights: base of one, lifted in the edit region and along mask edges.
    auto spatial_weights = at::ones_like(edit_mask);
    spatial_weights = spatial_weights + (edit_region_weight_multiplier - 1.0) * edit_mask;
    spatial_weights = spatial_weights + 0.5 * edge_magnitude;

    // Pixel-level reconstruction loss (spatially-weighted L1).
    auto pixel_diff = at::abs(predicted - target);
    auto pixel_loss = (pixel_diff * spatial_weights).mean();

    // Multi-scale perceptual features: each extractor consumes the previous
    // extractor's pooled output, with predicted and target carried in parallel
    // through the same weights.
    auto pred_feat0 = extract_features(
        predicted, extractor0_conv1_weight, extractor0_conv1_bias,
        extractor0_conv2_weight, extractor0_conv2_bias,
        stride1, pad1, pool_kernel, pool_stride);
    auto target_feat0 = extract_features(
        target, extractor0_conv1_weight, extractor0_conv1_bias,
        extractor0_conv2_weight, extractor0_conv2_bias,
        stride1, pad1, pool_kernel, pool_stride);

    auto pred_feat1 = extract_features(
        pred_feat0, extractor1_conv1_weight, extractor1_conv1_bias,
        extractor1_conv2_weight, extractor1_conv2_bias,
        stride1, pad1, pool_kernel, pool_stride);
    auto target_feat1 = extract_features(
        target_feat0, extractor1_conv1_weight, extractor1_conv1_bias,
        extractor1_conv2_weight, extractor1_conv2_bias,
        stride1, pad1, pool_kernel, pool_stride);

    auto pred_feat2 = extract_features(
        pred_feat1, extractor2_conv1_weight, extractor2_conv1_bias,
        extractor2_conv2_weight, extractor2_conv2_bias,
        stride1, pad1, pool_kernel, pool_stride);
    auto target_feat2 = extract_features(
        target_feat1, extractor2_conv1_weight, extractor2_conv1_bias,
        extractor2_conv2_weight, extractor2_conv2_bias,
        stride1, pad1, pool_kernel, pool_stride);

    const std::vector<torch::Tensor> pred_features{pred_feat0, pred_feat1, pred_feat2};
    const std::vector<torch::Tensor> target_features{target_feat0, target_feat1, target_feat2};

    // Perceptual loss summed over the three scales, each weighted by the
    // learnable per-scale magnitude.
    auto perceptual_loss = at::zeros({}, predicted.options());
    for (int scale_idx = 0; scale_idx < 3; ++scale_idx) {
        const auto& pred_feat = pred_features[scale_idx];
        const auto& target_feat = target_features[scale_idx];

        const std::vector<int64_t> feat_size{pred_feat.size(2), pred_feat.size(3)};
        auto downsampled_weights = at::upsample_bilinear2d(
            spatial_weights, /*output_size=*/feat_size, /*align_corners=*/false);

        auto feat_diff = at::abs(pred_feat - target_feat);
        auto scale_loss = (feat_diff * downsampled_weights).mean();
        scale_loss = scale_loss * at::abs(scale_weights[scale_idx]);
        perceptual_loss = perceptual_loss + scale_loss;
    }

    // Preservation loss for the non-edited region.
    auto inverse_mask = 1.0 - edit_mask;
    auto preservation_diff = at::abs(predicted - source) * inverse_mask;
    auto preservation_loss = preservation_diff.mean();

    auto result = pixel_loss_weight * pixel_loss
                + perceptual_loss_weight * perceptual_loss
                + 0.3 * preservation_loss;

    TORCH_CHECK(total_loss.dim() == 0,
                "total_loss must be a 0-dim scalar tensor, got ", total_loss.dim(), "D");
    TORCH_CHECK(result.sizes() == total_loss.sizes(),
                "result shape ", result.sizes(),
                " does not match pre-allocated ", total_loss.sizes());
    total_loss.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Edit consistency loss via pure ATen dispatch: a Sobel edge map of the "
          "edit mask drives spatial weights; a spatially-weighted L1 pixel term, a "
          "three-scale perceptual-feature term (cascaded conv-relu-conv-relu-pool "
          "extractors with bilinearly resized weights), and a preservation term "
          "over the non-edited region are combined into a scalar loss. No custom "
          "CUDA kernels. fp32; destination-passing style into a 0-dim output.");
}
