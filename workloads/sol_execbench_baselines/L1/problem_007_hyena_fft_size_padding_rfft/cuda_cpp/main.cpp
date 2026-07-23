#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Fused FFT size padding + real FFT for Hyena convolution.
//
// Reference (problem_007_hyena_fft_size_padding_rfft):
//     batch, channels, seqlen = x.shape
//     fft_size = 2 * seqlen
//     x_f32   = x.to(torch.float32)
//     x_freq  = torch.fft.rfft(x_f32, n=fft_size)   # dim=-1, no norm; implicit
//                                                    # zero-pad of x to fft_size
//     x_freq  = x_freq / fft_size                    # divide the COMPLEX tensor
//     x_freq_real = x_freq.real.contiguous()
//     x_freq_imag = x_freq.imag.contiguous()
//     return x_freq_real, x_freq_imag
//
// Per definition.json (insertion order):
//   x            : [batch_size, d_model, seqlen]            fp32
//   x_freq_real  : [batch_size, d_model, freq_len]          fp32  (DPS, pre-alloc)
//   x_freq_imag  : [batch_size, d_model, freq_len]          fp32  (DPS, pre-alloc)
//
//   freq_len = fft_size/2 + 1 = (2*seqlen)/2 + 1 = seqlen + 1.
//
// fft_size is derived from x's last dim (fft_size = 2*seqlen) exactly as the
// reference computes it, not passed as an input.
//
// Implementation: pure ATen dispatch -- NO custom CUDA kernels. The single
// at::fft_rfft call routes through cuFFT with the same fp32 precision profile
// the reference's torch.fft.rfft uses. at::fft_rfft returns a COMPLEX tensor;
// dividing it by the scalar fft_size before extracting real/imag matches the
// reference's "divide the complex tensor, then split" ordering exactly.
// at::real / at::imag return real-valued *views* into the complex storage, so
// each is materialised with .contiguous() before being copied into its
// pre-allocated DPS destination (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& x,
    torch::Tensor x_freq_real,
    torch::Tensor x_freq_imag
) {
    check_fp32_cuda(x, "x");
    check_fp32_cuda(x_freq_real, "x_freq_real");
    check_fp32_cuda(x_freq_imag, "x_freq_imag");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, channels, seqlen], got ", x.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(x.device());

    const int64_t seqlen = x.size(-1);
    const int64_t fft_size = 2 * seqlen;

    // x_freq = torch.fft.rfft(x_f32, n=fft_size)  -- dim defaults to -1, norm
    // defaults to None ("backward": no scaling on the forward transform). The
    // n=fft_size argument implicitly zero-pads x along dim=-1 up to fft_size.
    // Result is complex64 of last-dim length fft_size/2 + 1 = seqlen + 1.
    torch::Tensor x_freq = at::fft_rfft(
        x,
        /*n=*/fft_size,
        /*dim=*/-1,
        /*norm=*/c10::nullopt);

    // x_freq = x_freq / fft_size  -- normalize the COMPLEX tensor by the scalar
    // (scales both real and imaginary parts), matching the reference ordering.
    x_freq = x_freq / static_cast<double>(fft_size);

    // Split into real / imag. at::real / at::imag yield real-valued views into
    // the complex storage; .contiguous() materialises them as standalone fp32
    // tensors before the in-place copy into the DPS destinations.
    torch::Tensor real_part = at::real(x_freq).contiguous();
    torch::Tensor imag_part = at::imag(x_freq).contiguous();

    TORCH_CHECK(real_part.sizes() == x_freq_real.sizes(),
                "rfft real-part shape ", real_part.sizes(),
                " does not match pre-allocated x_freq_real ", x_freq_real.sizes());
    TORCH_CHECK(imag_part.sizes() == x_freq_imag.sizes(),
                "rfft imag-part shape ", imag_part.sizes(),
                " does not match pre-allocated x_freq_imag ", x_freq_imag.sizes());

    x_freq_real.copy_(real_part);
    x_freq_imag.copy_(imag_part);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Hyena FFT-size padding + real FFT via at::fft_rfft (cuFFT): "
          "rfft(x, n=2*seqlen, dim=-1) / fft_size, split into real/imag; "
          "fp32; destination-passing style.");
}
