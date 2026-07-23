#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Hyena order-2 gating + FFT convolution + output projection.
//
// Reference (052_seqlen-finetuned-reconstructed_hyena_gating_and_output_projection):
//     seqlen   = v.shape[-1]
//     fft_size = 2 * seqlen
//     v = v * x1                                   # order-2 gate (dropout=0)
//     k_0   = k[0]                                 # [d_model, L]
//     bias_0= bias[0]                              # [d_model]
//     k_f = rfft(k_0, n=fft_size) / fft_size       # complex, divide complex tensor
//     u_f = rfft(v,   n=fft_size)                  # complex
//     y   = irfft(u_f * k_f.unsqueeze(0), n=fft_size, norm='forward')[..., :seqlen]
//     v   = y + v * bias_0.unsqueeze(-1)           # gated v reused here
//     y   = v * x0                                 # final gate
//     y   = y.transpose(1, 2)                      # [B, d_model, L] -> [B, L, d_model]
//     output = matmul(y, out_proj_weight.t()) + out_proj_bias
//
// All math is float32; the FFT intermediates are complex64. rfft / irfft default
// to dim=-1; the forward rfft uses the default norm (None, i.e. "backward": no
// forward scaling) while the inverse uses norm='forward' (which divides by N on
// the inverse). The reference's manual k_f / fft_size division is therefore the
// only forward-side scaling and is kept as an explicit divide of the COMPLEX
// tensor. The n=fft_size argument implicitly zero-pads along dim=-1 to fft_size,
// and the post-irfft [..., :seqlen] slice keeps the causal head.
//
// Per definition.json (insertion order):
//   inputs (all float32):
//     v               : [batch_size, d_model, seq_len]
//     x0              : [batch_size, d_model, seq_len]
//     x1              : [batch_size, d_model, seq_len]
//     k               : [order_minus_1, d_model, seq_len]
//     bias            : [order_minus_1, d_model]
//     out_proj_weight : [d_model, d_model]
//     out_proj_bias   : [d_model]
//   output (float32, DPS pre-alloc, written in place):
//     output          : [batch_size, seq_len, d_model]
//
// Implementation: pure ATen dispatch -- no custom CUDA kernels. at::fft_rfft /
// at::fft_irfft route through cuFFT at the same fp32 precision the reference's
// torch.fft uses; the complex multiply, gating, transpose, and output linear
// match the reference exactly. Destination-passing style.

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& v,
    const torch::Tensor& x0,
    const torch::Tensor& x1,
    const torch::Tensor& k,
    const torch::Tensor& bias,
    const torch::Tensor& out_proj_weight,
    const torch::Tensor& out_proj_bias,
    torch::Tensor output
) {
    check_fp32_cuda(v, "v");
    check_fp32_cuda(x0, "x0");
    check_fp32_cuda(x1, "x1");
    check_fp32_cuda(k, "k");
    check_fp32_cuda(bias, "bias");
    check_fp32_cuda(out_proj_weight, "out_proj_weight");
    check_fp32_cuda(out_proj_bias, "out_proj_bias");

    TORCH_CHECK(v.dim() == 3, "v must be 3D [batch, d_model, seqlen], got ", v.dim(), "D");
    TORCH_CHECK(k.dim() == 3, "k must be 3D [order_minus_1, d_model, seqlen], got ", k.dim(), "D");
    TORCH_CHECK(bias.dim() == 2, "bias must be 2D [order_minus_1, d_model], got ", bias.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(v.device());

    const int64_t seqlen = v.size(-1);
    const int64_t fft_size = 2 * seqlen;

    // Order-2 gate (dropout=0.0, so the multiply is the whole step).
    auto v_gated = v * x1;

    // Filter k[0] and skip-connection bias[0].
    auto k_0 = k.select(/*dim=*/0, 0);      // [d_model, seqlen]
    auto bias_0 = bias.select(/*dim=*/0, 0);  // [d_model]

    // rfft along dim=-1 with default norm (None / "backward"); the n=fft_size
    // argument zero-pads to fft_size. The filter spectrum is divided by fft_size
    // as a COMPLEX-tensor scale, matching the reference ordering.
    auto k_f = at::fft_rfft(k_0, /*n=*/fft_size, /*dim=*/-1, /*norm=*/c10::nullopt)
                   / static_cast<double>(fft_size);   // [d_model, fft_size/2+1]
    auto u_f = at::fft_rfft(v_gated, /*n=*/fft_size, /*dim=*/-1, /*norm=*/c10::nullopt);
                                                       // [batch, d_model, fft_size/2+1]

    // Frequency-domain multiply (broadcast k_f over batch) then inverse rfft with
    // norm='forward' (the 1/N inverse scaling lives here); keep the causal head.
    auto prod = u_f * k_f.unsqueeze(0);
    auto y_conv = at::fft_irfft(prod, /*n=*/fft_size, /*dim=*/-1, /*norm=*/c10::optional<c10::string_view>("forward"));
    y_conv = y_conv.narrow(/*dim=*/-1, /*start=*/0, /*length=*/seqlen);

    // Skip connection with bias (uses the gated v, not the original input).
    auto v_skip = y_conv + v_gated * bias_0.unsqueeze(-1);

    // Final gate with x0.
    auto y_gated = v_skip * x0;

    // [batch, d_model, L] -> [batch, L, d_model].
    auto y_t = y_gated.transpose(1, 2);

    // Output projection: y @ out_proj_weight.T + out_proj_bias.
    auto result = at::matmul(y_t, out_proj_weight.t()) + out_proj_bias;

    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());

    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Hyena order-2 gating + FFT convolution + output projection via ATen / "
          "cuFFT: v*x1 gate, rfft(n=2L)/N filter spectrum, frequency-domain "
          "multiply, irfft(norm='forward')[:L], bias skip, x0 gate, transpose, "
          "output linear. fp32 throughout; destination-passing style.");
}
