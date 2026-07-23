#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Joint attention context projection (Stable Diffusion 3.5 joint attention).
//
// Reference (problem_081_joint_attention_context_projection):
//
//     batch_size      = image_attention_output.shape[0]
//     image_seq_len   = image_attention_output.shape[1]
//     context_seq_len = context_attention_output.shape[1]
//
//     # 1. concat along sequence dim -> [B, image_seq_len + context_seq_len, inner_dim]
//     combined_attention = torch.cat([image_attention_output,
//                                     context_attention_output], dim=1)
//
//     # 2. duplicate along feature dim -> [B, S, 2*inner_dim]
//     combined_features  = torch.cat([combined_attention,
//                                     combined_attention], dim=-1)
//
//     # 3. linear: x @ W^T + b   (W: [inner_dim, 2*inner_dim], b: [inner_dim])
//     projected_output   = torch.matmul(combined_features, to_out_weight.t()) + to_out_bias
//
//     # 4. split back along sequence dim
//     projected_image    = projected_output[:, :image_seq_len, :]
//     projected_context  = projected_output[:, image_seq_len:, :]
//     return projected_image, projected_context
//
// Per definition.json (insertion order):
//   image_attention_output   : [batch_size, image_seq_len,   inner_dim]      fp32
//   context_attention_output : [batch_size, context_seq_len, inner_dim]      fp32
//   to_out_weight            : [inner_dim, proj_input_dim]  (proj_input_dim = inner_dim*2)  fp32
//   to_out_bias              : [inner_dim]                                   fp32
//   projected_image          : [batch_size, image_seq_len,   inner_dim]      fp32  (DPS)
//   projected_context        : [batch_size, context_seq_len, inner_dim]      fp32  (DPS)
//
// Subtleties faithfully reproduced:
//   * Feature duplication: combined_features = cat([combined, combined], dim=-1)
//     doubles the feature width from inner_dim to proj_input_dim = 2*inner_dim,
//     so it matches to_out_weight's column count. We reproduce the literal
//     cat-of-two-copies rather than the algebraic shortcut
//     (x @ W1^T + x @ W2^T), so the matmul reduction order / FP rounding is the
//     same single GEMM the reference issues.
//   * proj_input_dim is the DOUBLED inner_dim (= inner_dim * 2 per the axes
//     "expr"), NOT a free axis. to_out_weight is [inner_dim, 2*inner_dim].
//   * Linear is x @ W.t() + b (PyTorch F.linear convention written out via
//     matmul). at::matmul + at::transpose route through the same cuBLAS GEMM
//     the reference torch.matmul issues, so the result is bit-matched.
//   * Sequence split: the first image_seq_len rows go to projected_image, the
//     remaining context_seq_len rows go to projected_context. We .contiguous()
//     each slice before copy_ so the DPS write into the (contiguous)
//     pre-allocated outputs is well-defined.
//
// Implementation: pure ATen tensor ops -- NO custom CUDA kernels. Outputs are
// written in place into the pre-allocated destinations (destination-passing
// style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& image_attention_output,
    const torch::Tensor& context_attention_output,
    const torch::Tensor& to_out_weight,
    const torch::Tensor& to_out_bias,
    torch::Tensor projected_image,
    torch::Tensor projected_context
) {
    check_fp32_cuda(image_attention_output, "image_attention_output");
    check_fp32_cuda(context_attention_output, "context_attention_output");
    check_fp32_cuda(to_out_weight, "to_out_weight");
    check_fp32_cuda(to_out_bias, "to_out_bias");
    check_fp32_cuda(projected_image, "projected_image");
    check_fp32_cuda(projected_context, "projected_context");
    TORCH_CHECK(image_attention_output.dim() == 3,
                "image_attention_output must be 3D [B, image_seq_len, inner_dim], got ",
                image_attention_output.dim(), "D");
    TORCH_CHECK(context_attention_output.dim() == 3,
                "context_attention_output must be 3D [B, context_seq_len, inner_dim], got ",
                context_attention_output.dim(), "D");
    TORCH_CHECK(to_out_weight.dim() == 2,
                "to_out_weight must be 2D [inner_dim, proj_input_dim], got ",
                to_out_weight.dim(), "D");
    TORCH_CHECK(to_out_bias.dim() == 1,
                "to_out_bias must be 1D [inner_dim], got ", to_out_bias.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(image_attention_output.device());

    const int64_t image_seq_len = image_attention_output.size(1);

    // 1. Concatenate along the sequence dim (dim=1):
    //    [B, image_seq_len + context_seq_len, inner_dim]
    torch::Tensor combined_attention =
        at::cat({image_attention_output, context_attention_output}, /*dim=*/1);

    // 2. Duplicate along the feature dim (dim=-1):
    //    [B, S, 2*inner_dim] == [B, S, proj_input_dim]
    torch::Tensor combined_features =
        at::cat({combined_attention, combined_attention}, /*dim=*/-1);

    // 3. Linear projection: combined_features @ to_out_weight.t() + to_out_bias
    //    [B, S, proj_input_dim] @ [proj_input_dim, inner_dim] -> [B, S, inner_dim]
    torch::Tensor projected_output =
        at::matmul(combined_features, to_out_weight.t()) + to_out_bias;

    // 4. Split back along the sequence dim.
    torch::Tensor img_slice =
        projected_output.slice(/*dim=*/1, /*start=*/0, /*end=*/image_seq_len).contiguous();
    torch::Tensor ctx_slice =
        projected_output.slice(/*dim=*/1, /*start=*/image_seq_len).contiguous();

    TORCH_CHECK(img_slice.sizes() == projected_image.sizes(),
                "computed projected_image shape ", img_slice.sizes(),
                " does not match pre-allocated output ", projected_image.sizes());
    TORCH_CHECK(ctx_slice.sizes() == projected_context.sizes(),
                "computed projected_context shape ", ctx_slice.sizes(),
                " does not match pre-allocated output ", projected_context.sizes());

    projected_image.copy_(img_slice);
    projected_context.copy_(ctx_slice);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Joint attention context projection via ATen (cat + cat + matmul + "
          "bias + split); fp32; destination-passing style.");
}
