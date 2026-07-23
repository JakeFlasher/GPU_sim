#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>
#include <vector>

// Fused gated-projection layer with causal 1D convolution. The dominant
// non-attention op in the LFM2 architecture: triple linear projection ->
// element-wise gating -> grouped causal conv1d -> output gating -> output
// projection.
//
// Reference (problem_005_conv_gated_projection_with_causal_conv):
//   batch_size, seq_len, hidden_size = x.shape
//   conv_kernel_size = conv_weight.shape[2]
//   # 1. Triple linear projection -> (B, L, 3*H)
//   BCx = F.linear(x, in_proj_weight, in_proj_bias)
//   # 2. Transpose for conv1d -> (B, 3*H, L)
//   BCx = BCx.transpose(-1, -2)
//   # 3. Split into B, C, x_proj along the channel dim -> 3x (B, H, L)
//   B, C, x_proj = BCx.chunk(3, dim=1)
//   # 4. Element-wise gating -> (B, H, L)
//   Bx = B * x_proj
//   # 5. Left-pad causal (kernel_size - 1 on the left)
//   Bx_padded = F.pad(Bx, (conv_kernel_size - 1, 0))
//   # 6. Depthwise (groups=hidden_size) causal conv1d
//   conv_out = F.conv1d(Bx_padded, conv_weight, conv_bias, groups=hidden_size)
//   # 7. Output gating with C -> (B, H, L)
//   y = C * conv_out
//   # 8. Transpose back -> (B, L, H), make contiguous
//   y = y.transpose(-1, -2).contiguous()
//   # 9. Final output projection -> (B, L, H)
//   output = F.linear(y, out_proj_weight, out_proj_bias)
//   return output
//
// Per definition.json (insertion order):
//   x               : [batch_size, seq_len, hidden_size]     bf16
//   in_proj_weight  : [triple_hidden, hidden_size]           bf16   (triple_hidden = 3*H)
//   in_proj_bias    : [triple_hidden]                        bf16
//   conv_weight     : [hidden_size, 1, conv_kernel_size]     bf16   (depthwise)
//   conv_bias       : [hidden_size]                          bf16
//   out_proj_weight : [hidden_size, hidden_size]             bf16
//   out_proj_bias   : [hidden_size]                          bf16
//   output          : [batch_size, seq_len, hidden_size]     bf16   (DPS, pre-alloc)
//
// Notes on op-for-op fidelity:
//   * hidden_size (H) is derived from x.size(2); conv_kernel_size from
//     conv_weight.size(2). No constant (2048 / 4) is hard-coded -- both are read
//     off the actual tensor shapes. The depthwise conv uses groups == H.
//   * chunk(3, dim=1) on a 3*H-channel tensor (after transpose) yields three
//     contiguous H-channel slices, IN ORDER: B (channels [0, H)), C
//     (channels [H, 2H)), x_proj (channels [2H, 3H)). The FIRST chunk (B) and
//     the THIRD chunk (x_proj) form the pre-conv gate (Bx = B * x_proj); the
//     SECOND chunk (C) gates the conv output (y = C * conv_out). Implemented
//     with narrow on dim 1.
//   * Causal left pad: constant_pad_nd pads the LAST (length) dim;
//     {conv_kernel_size - 1, 0} pads (conv_kernel_size - 1) zeros on the left,
//     0 on the right -> output length stays seq_len after the valid (padding=0)
//     depthwise conv1d.
//   * The pre-projection transpose / chunk views are non-contiguous; ATen's
//     elementwise mul handles strided operands. The reference explicitly calls
//     .contiguous() on y before the final linear -- reproduced here so the
//     final F.linear sees the same contiguous (B, L, H) layout.
//   * All math stays in bf16; ATen's linear / conv1d accumulate internally and
//     emit bf16 outputs, matching the reference's bf16 cast profile (no
//     intermediate upcast in run()).
//
// Implementation: pure ATen ops -- NO custom CUDA kernels. Output is written in
// place into the pre-allocated destination (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& x,
    const torch::Tensor& in_proj_weight,
    const torch::Tensor& in_proj_bias,
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& out_proj_weight,
    const torch::Tensor& out_proj_bias,
    torch::Tensor output
) {
    check_bf16_cuda(x, "x");
    check_bf16_cuda(in_proj_weight, "in_proj_weight");
    check_bf16_cuda(in_proj_bias, "in_proj_bias");
    check_bf16_cuda(conv_weight, "conv_weight");
    check_bf16_cuda(conv_bias, "conv_bias");
    check_bf16_cuda(out_proj_weight, "out_proj_weight");
    check_bf16_cuda(out_proj_bias, "out_proj_bias");
    check_bf16_cuda(output, "output");

    TORCH_CHECK(x.dim() == 3,
                "x must be 3D [B, seq, hidden], got ", x.dim(), "D");
    TORCH_CHECK(conv_weight.dim() == 3,
                "conv_weight must be 3D [H, 1, k], got ", conv_weight.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(x.device());

    // hidden_size (H) and conv kernel size derived from the tensor shapes.
    const int64_t hidden_size = x.size(2);
    const int64_t conv_kernel_size = conv_weight.size(2);

    // 1. Triple linear projection: (B, L, H) -> (B, L, 3*H).
    torch::Tensor BCx = at::linear(x, in_proj_weight, in_proj_bias);

    // 2. Transpose for conv1d: (B, L, 3*H) -> (B, 3*H, L).
    BCx = BCx.transpose(-1, -2);

    // 3. Split into B, C, x_proj along the channel dim (chunk(3, dim=1)):
    //    first H channels -> B, second H -> C, third H -> x_proj.
    torch::Tensor B_gate = BCx.narrow(/*dim=*/1, /*start=*/0, hidden_size);
    torch::Tensor C_gate = BCx.narrow(/*dim=*/1, /*start=*/hidden_size, hidden_size);
    torch::Tensor x_proj = BCx.narrow(/*dim=*/1, /*start=*/2 * hidden_size, hidden_size);

    // 4. Element-wise gating: Bx = B * x_proj.
    torch::Tensor Bx = B_gate * x_proj;

    // 5. Causal left pad: pad the last (length) dim by (conv_kernel_size - 1)
    //    zeros on the left, 0 on the right.
    const std::vector<int64_t> pad_lr{conv_kernel_size - 1, 0};
    torch::Tensor Bx_padded = at::constant_pad_nd(Bx, /*pad=*/pad_lr, /*value=*/0);

    // 6. Depthwise (groups=hidden_size) causal conv1d on the padded input.
    //    Explicitly-typed int64 vectors so the IntArrayRef conv1d overload is
    //    the unique match (a braced-init {0} is ambiguous against the
    //    string_view padding overload when bias is passed as a raw Tensor).
    const std::vector<int64_t> stride{1}, padding{0}, dilation{1};
    torch::Tensor conv_out = at::conv1d(
        Bx_padded,
        conv_weight,
        conv_bias,
        /*stride=*/stride,
        /*padding=*/padding,
        /*dilation=*/dilation,
        /*groups=*/hidden_size);

    // 7. Output gating with C: y = C * conv_out.
    torch::Tensor y = C_gate * conv_out;

    // 8. Transpose back to (B, L, H) and make contiguous (matches reference).
    y = y.transpose(-1, -2).contiguous();

    // 9. Final output projection: (B, L, H) -> (B, L, H).
    torch::Tensor result = at::linear(y, out_proj_weight, out_proj_bias);

    TORCH_CHECK(result.sizes() == output.sizes(),
                "fused layer output shape ", result.sizes(),
                " does not match pre-allocated output ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Gated-projection layer with causal depthwise conv1d via ATen "
          "(linear / gating / causal conv1d / output projection; no custom CUDA "
          "kernels); bf16; destination-passing style.");
}
