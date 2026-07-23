#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <string>

// Fused Mamba conv1d with gating.
//
// Reference (problem_029_mamba_conv1d_with_gating):
//   batch_size, seq_len, _ = hidden_states.shape
//   intermediate_size = 16384 ; conv_kernel_size = 4
//
//   # 1. Gated linear projection: (B, L, H) -> (B, L, 2*I)
//   projected_states = F.linear(hidden_states, in_proj_weight, in_proj_bias)
//   # 2. Transpose for conv1d: (B, L, 2*I) -> (B, 2*I, L)
//   projected_states = projected_states.transpose(1, 2)
//   # 3. Split into hidden states and gate: (B, 2*I, L) -> 2x (B, I, L)
//   hidden_states_conv, gate = projected_states.chunk(2, dim=1)
//   # 4. Apply attention mask before convolution
//   hidden_states_conv = hidden_states_conv * attention_mask.unsqueeze(1)
//   # 5. Causal 1D depthwise convolution (left-pad by conv_kernel_size - 1)
//   hidden_states_padded = F.pad(hidden_states_conv, (conv_kernel_size - 1, 0))
//   hidden_states_conv = F.conv1d(hidden_states_padded, conv1d_weight,
//                                 conv1d_bias, groups=intermediate_size)
//   # 6. SiLU activation: silu(x) = x * sigmoid(x)
//   hidden_states_conv = hidden_states_conv * torch.sigmoid(hidden_states_conv)
//   # 7. Apply attention mask after convolution
//   hidden_states_conv = hidden_states_conv * attention_mask.unsqueeze(1)
//   return hidden_states_conv, gate
//
// Per definition.json (insertion order):
//   hidden_states  : [B, seq, hidden]            bf16
//   attention_mask : [B, seq]                     bf16
//   in_proj_weight : [proj_out_size, hidden]      bf16   (proj_out_size = 2*I)
//   in_proj_bias   : [proj_out_size]              bf16
//   conv1d_weight  : [intermediate, 1, conv_k]    bf16   (depthwise)
//   conv1d_bias    : [intermediate]               bf16
//   output_hidden_states : [B, intermediate, seq] bf16   (DPS, pre-alloc)
//   gate                 : [B, intermediate, seq] bf16   (DPS, pre-alloc)
//
// Notes on op-for-op fidelity:
//   * intermediate_size is derived from conv1d_weight.size(0) (== I); the gate
//     / output channel count is I, the projection output is 2*I. No constant
//     16384 is hard-coded -- it is read off the actual tensor shapes.
//   * chunk(2, dim=1) of a 2*I-channel tensor yields the first I channels as
//     hidden_states_conv and the second I channels as gate (narrow on dim 1).
//   * attention_mask.unsqueeze(1): mask is [B, seq] -> [B, 1, seq], broadcasts
//     across the I channel dim. Applied BOTH before the conv (step 4) and after
//     SiLU (step 7).
//   * Causal left pad: constant_pad_nd pads the LAST dim; {conv_k - 1, 0} pads
//     (conv_kernel_size - 1) zeros on the left, 0 on the right -> output length
//     stays seq_len after the valid (padding=0) depthwise conv1d.
//   * Depthwise conv: groups == intermediate_size, so each of the I channels is
//     convolved by its own length-conv_k filter (conv1d_weight is [I, 1, k]).
//   * SiLU via at::silu == x * sigmoid(x), the fused equivalent of the
//     reference's explicit x * torch.sigmoid(x). All math stays in bf16; ATen's
//     linear / conv1d accumulate internally and emit bf16 outputs, matching the
//     reference's bf16 cast profile (no intermediate upcast in run()).
//
// Implementation: pure ATen ops -- NO custom CUDA kernels. Outputs are written
// in place into the pre-allocated destinations (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& attention_mask,
    const torch::Tensor& in_proj_weight,
    const torch::Tensor& in_proj_bias,
    const torch::Tensor& conv1d_weight,
    const torch::Tensor& conv1d_bias,
    torch::Tensor output_hidden_states,
    torch::Tensor gate
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(attention_mask, "attention_mask");
    check_bf16_cuda(in_proj_weight, "in_proj_weight");
    check_bf16_cuda(in_proj_bias, "in_proj_bias");
    check_bf16_cuda(conv1d_weight, "conv1d_weight");
    check_bf16_cuda(conv1d_bias, "conv1d_bias");
    check_bf16_cuda(output_hidden_states, "output_hidden_states");
    check_bf16_cuda(gate, "gate");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, seq, hidden], got ", hidden_states.dim(), "D");
    TORCH_CHECK(attention_mask.dim() == 2,
                "attention_mask must be 2D [B, seq], got ", attention_mask.dim(), "D");
    TORCH_CHECK(conv1d_weight.dim() == 3,
                "conv1d_weight must be 3D [I, 1, k], got ", conv1d_weight.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    // intermediate_size (I) and conv kernel size derived from conv1d_weight.
    const int64_t intermediate_size = conv1d_weight.size(0);
    const int64_t conv_kernel_size = conv1d_weight.size(2);

    // 1. Gated linear projection: (B, L, H) -> (B, L, 2*I)
    torch::Tensor projected_states = at::linear(hidden_states, in_proj_weight, in_proj_bias);

    // 2. Transpose for conv1d: (B, L, 2*I) -> (B, 2*I, L)
    projected_states = projected_states.transpose(1, 2);

    // 3. Split into hidden states and gate: (B, 2*I, L) -> 2x (B, I, L)
    //    chunk(2, dim=1): first I channels -> conv path, second I channels -> gate.
    torch::Tensor hidden_states_conv = projected_states.narrow(/*dim=*/1, /*start=*/0, intermediate_size);
    torch::Tensor gate_result = projected_states.narrow(/*dim=*/1, /*start=*/intermediate_size, intermediate_size);

    // 4. Apply attention mask before convolution: mask [B, seq] -> [B, 1, seq].
    torch::Tensor mask = attention_mask.unsqueeze(1);
    hidden_states_conv = hidden_states_conv * mask;

    // 5. Causal 1D depthwise convolution. Left-pad the last (length) dim by
    //    (conv_kernel_size - 1) and run a valid (padding=0) grouped conv1d.
    const std::vector<int64_t> pad_lr{conv_kernel_size - 1, 0};
    torch::Tensor hidden_states_padded =
        at::constant_pad_nd(hidden_states_conv, /*pad=*/pad_lr, /*value=*/0);
    // Explicitly-typed int64 vectors so the IntArrayRef conv1d overload is the
    // unique match (a braced-init {0} is ambiguous against the string_view
    // padding overload when bias is passed as a raw Tensor).
    const std::vector<int64_t> stride{1}, padding{0}, dilation{1};
    hidden_states_conv = at::conv1d(
        hidden_states_padded,
        conv1d_weight,
        conv1d_bias,
        /*stride=*/stride,
        /*padding=*/padding,
        /*dilation=*/dilation,
        /*groups=*/intermediate_size);

    // 6. SiLU activation: silu(x) = x * sigmoid(x).
    hidden_states_conv = at::silu(hidden_states_conv);

    // 7. Apply attention mask after convolution.
    hidden_states_conv = hidden_states_conv * mask;

    TORCH_CHECK(hidden_states_conv.sizes() == output_hidden_states.sizes(),
                "output_hidden_states shape ", hidden_states_conv.sizes(),
                " does not match pre-allocated output ", output_hidden_states.sizes());
    TORCH_CHECK(gate_result.sizes() == gate.sizes(),
                "gate shape ", gate_result.sizes(),
                " does not match pre-allocated output ", gate.sizes());

    output_hidden_states.copy_(hidden_states_conv);
    gate.copy_(gate_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Mamba conv1d with gating via ATen (linear / causal depthwise conv1d / "
          "SiLU / masking; no custom CUDA kernels); bf16; destination-passing style.");
}
