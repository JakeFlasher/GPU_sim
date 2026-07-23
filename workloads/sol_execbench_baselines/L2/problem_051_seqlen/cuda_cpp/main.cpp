#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <string>
#include <vector>

// Complete Hyena forward block: residual + LayerNorm, the full Hyena operator
// (input projection, short depthwise conv, implicit-filter MLP, order-2 FFT
// convolution with gating), a second LayerNorm, a GELU MLP, and the final
// residual addition.
//
// Reference (051_seqlen-finetuned-reconstructed_hyena_complete_forward_block),
// op-for-op:
//   residual = hidden_states.float()
//   normed   = layer_norm1(residual) * norm1_weight + norm1_bias
//   u  = F.linear(normed, in_proj_weight, in_proj_bias).transpose(1, 2)
//   uc = F.conv1d(F.pad(u, (2, 2)), short_conv_weight, short_conv_bias,
//                 groups=inner_width)[..., :l_filter]
//   x = uc.split(d_model, dim=1)[:-1]; v = uc.split(d_model, dim=1)[-1]
//   # implicit filter: positional features -> sinusoidal-activation MLP
//   t          = linspace(0, 1, l_filter)[None, :, None]
//   t_rescaled = linspace(0, l_filter - 1, l_filter)[None, :, None]
//   w = 2*pi*t_rescaled / l_filter ; f = linspace(1e-4, 1, 2)[None, None]
//   z = cat([t, cos(-f*w), sin(-f*w)], dim=-1)            # last dim == emb_dim
//   h = filter MLP over z with sin(sin_freq * .) activations (final linear no bias)
//   decay = exp(-t * exp_mod_deltas.abs()) ; h = h * (decay + exp_mod_shift)
//   h = h + filter_bias.view(1, 1, d_model)
//   k = h.transpose(0, 1).reshape(1, d_model, l_filter)
//   bias_reshaped = filter_bias.reshape(1, d_model)
//   for o, x_i in enumerate(reversed(x[1:])):            # order-2 -> one step
//       v = v * x_i
//       fft_size = 2 * l_filter
//       k_f = rfft(k[o].float(), n=fft_size) / fft_size
//       v_f = rfft(v.float(),    n=fft_size)
//       y   = irfft(v_f * k_f, n=fft_size, norm='forward')[..., :l_filter]
//       v   = y + v * bias_reshaped[o].unsqueeze(-1)
//   y = (v * x[0]).transpose(1, 2)
//   if l_filter < seq_len: y = F.pad(y, (0, 0, 0, seq_len - l_filter))
//   hyena_out = F.linear(y, out_proj_weight, out_proj_bias)
//   residual  = hyena_out + residual
//   normed    = layer_norm2(residual) * norm2_weight + norm2_bias
//   mlp_out   = F.linear(F.gelu(F.linear(normed, mlp_fc1_weight, mlp_fc1_bias),
//                               approximate='tanh'), mlp_fc2_weight, mlp_fc2_bias)
//   output    = mlp_out + residual
//
// Per definition.json (insertion order), all tensors float32:
//   hidden_states               : [batch, seq_len, d_model=256]
//   norm1_weight / norm1_bias   : [d_model]
//   norm2_weight / norm2_bias   : [d_model]
//   in_proj_weight              : [inner_width=768, d_model]
//   in_proj_bias                : [inner_width]
//   short_conv_weight           : [inner_width, 1, short_filter_order=3]
//   short_conv_bias             : [inner_width]
//   filter_linear1_weight       : [filter_order=64, emb_dim=5]
//   filter_linear1_bias         : [filter_order]
//   sin_freq                    : [1, filter_order]
//   filter_linear2_weight       : [filter_order, filter_order]
//   filter_linear2_bias         : [filter_order]
//   filter_linear3_weight       : [filter_order, filter_order]
//   filter_linear3_bias         : [filter_order]
//   filter_linear_final_weight  : [d_model, filter_order]   (no bias)
//   filter_bias                 : [d_model]
//   exp_mod_deltas              : [1, 1, d_model]
//   out_proj_weight             : [d_model, d_model]
//   out_proj_bias               : [d_model]
//   mlp_fc1_weight              : [d_inner=1024, d_model]
//   mlp_fc1_bias                : [d_inner]
//   mlp_fc2_weight              : [d_model, d_inner]
//   mlp_fc2_bias                : [d_model]
//   layer_norm_eps              : scalar  (passed as double)
//   exp_mod_shift               : scalar  (passed as double)
//   output                      : [batch, seq_len, d_model]  (float32, DPS)
//
// Constants are fixed by the workload table: d_model=256, order=2,
// l_max=32768, inner_width = d_model * (order + 1) = 768. The order loop is a
// fixed static range (order - 1 = 1 iteration here), not a data-dependent
// scan. The two positional-feature bands (bands=2) make z's last dimension
// 1 + 2 + 2 = 5, matching emb_dim.
//
// Precision: everything is float32, including the FFT path (the rfft / irfft
// intermediates are complex64). at::fft_rfft / at::fft_irfft route through
// cuFFT at the same fp32 precision the reference's torch.fft uses. The forward
// rfft uses the default norm (None / "backward": no forward scaling) and the
// filter spectrum is divided by fft_size as a COMPLEX-tensor scale, exactly as
// the reference orders it; the inverse irfft carries norm='forward' (its 1/N
// scaling). The post-irfft [..., :l_filter] slice keeps the causal head. The
// two LayerNorms use unbiased=False variance and exact-erf normalisation; the
// MLP GELU uses the tanh approximation (approximate='tanh').
//
// Implementation: pure ATen / cuFFT / cuDNN / cuBLAS dispatch -- no custom CUDA
// kernels. Output is written in place into the pre-allocated destination
// (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

// LayerNorm over the last dimension with population (unbiased=False) variance,
// then the affine fold weight * . + bias. Everything stays fp32.
static torch::Tensor layer_norm_fp32(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps
) {
    auto mean = x.mean(/*dim=*/{-1}, /*keepdim=*/true);
    auto var = x.var(/*dim=*/{-1}, /*unbiased=*/false, /*keepdim=*/true);
    auto normed = (x - mean) / torch::sqrt(var + eps);
    return normed * weight + bias;
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& norm1_bias,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& norm2_bias,
    const torch::Tensor& in_proj_weight,
    const torch::Tensor& in_proj_bias,
    const torch::Tensor& short_conv_weight,
    const torch::Tensor& short_conv_bias,
    const torch::Tensor& filter_linear1_weight,
    const torch::Tensor& filter_linear1_bias,
    const torch::Tensor& sin_freq,
    const torch::Tensor& filter_linear2_weight,
    const torch::Tensor& filter_linear2_bias,
    const torch::Tensor& filter_linear3_weight,
    const torch::Tensor& filter_linear3_bias,
    const torch::Tensor& filter_linear_final_weight,
    const torch::Tensor& filter_bias,
    const torch::Tensor& exp_mod_deltas,
    const torch::Tensor& out_proj_weight,
    const torch::Tensor& out_proj_bias,
    const torch::Tensor& mlp_fc1_weight,
    const torch::Tensor& mlp_fc1_bias,
    const torch::Tensor& mlp_fc2_weight,
    const torch::Tensor& mlp_fc2_bias,
    double layer_norm_eps,
    double exp_mod_shift,
    torch::Tensor output
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(norm1_weight, "norm1_weight");
    check_fp32_cuda(norm1_bias, "norm1_bias");
    check_fp32_cuda(norm2_weight, "norm2_weight");
    check_fp32_cuda(norm2_bias, "norm2_bias");
    check_fp32_cuda(in_proj_weight, "in_proj_weight");
    check_fp32_cuda(in_proj_bias, "in_proj_bias");
    check_fp32_cuda(short_conv_weight, "short_conv_weight");
    check_fp32_cuda(short_conv_bias, "short_conv_bias");
    check_fp32_cuda(filter_linear1_weight, "filter_linear1_weight");
    check_fp32_cuda(filter_linear1_bias, "filter_linear1_bias");
    check_fp32_cuda(sin_freq, "sin_freq");
    check_fp32_cuda(filter_linear2_weight, "filter_linear2_weight");
    check_fp32_cuda(filter_linear2_bias, "filter_linear2_bias");
    check_fp32_cuda(filter_linear3_weight, "filter_linear3_weight");
    check_fp32_cuda(filter_linear3_bias, "filter_linear3_bias");
    check_fp32_cuda(filter_linear_final_weight, "filter_linear_final_weight");
    check_fp32_cuda(filter_bias, "filter_bias");
    check_fp32_cuda(exp_mod_deltas, "exp_mod_deltas");
    check_fp32_cuda(out_proj_weight, "out_proj_weight");
    check_fp32_cuda(out_proj_bias, "out_proj_bias");
    check_fp32_cuda(mlp_fc1_weight, "mlp_fc1_weight");
    check_fp32_cuda(mlp_fc1_bias, "mlp_fc1_bias");
    check_fp32_cuda(mlp_fc2_weight, "mlp_fc2_weight");
    check_fp32_cuda(mlp_fc2_bias, "mlp_fc2_bias");
    check_fp32_cuda(output, "output");
    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [batch, seq_len, d_model], got ",
                hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t d_model = 256;
    constexpr int64_t order = 2;
    constexpr int64_t l_max = 32768;
    constexpr int64_t inner_width = d_model * (order + 1);
    constexpr int64_t bands = 2;
    const double pi = 3.14159265358979323846;

    const int64_t seq_len = hidden_states.size(1);
    const int64_t l_filter = std::min<int64_t>(seq_len, l_max);
    const auto device = hidden_states.device();
    const auto fopts = at::TensorOptions().dtype(torch::kFloat32).device(device);

    // First residual + LayerNorm.
    auto residual = hidden_states.to(torch::kFloat32);
    auto normed = layer_norm_fp32(residual, norm1_weight, norm1_bias, layer_norm_eps);

    // Hyena input projection, then move channels to the conv axis.
    auto u = at::linear(normed, in_proj_weight, in_proj_bias).transpose(1, 2);

    // Short depthwise convolution: pad both ends by short_filter_order - 1 == 2,
    // grouped over all inner_width channels, then keep the causal head.
    auto u_padded = at::constant_pad_nd(u, /*pad=*/std::vector<int64_t>{2, 2}, /*value=*/0);
    auto uc = at::conv1d(u_padded, short_conv_weight, short_conv_bias,
                         /*stride=*/std::vector<int64_t>{1},
                         /*padding=*/std::vector<int64_t>{0},
                         /*dilation=*/std::vector<int64_t>{1},
                         /*groups=*/inner_width);
    uc = uc.narrow(/*dim=*/-1, /*start=*/0, /*length=*/l_filter);

    // Split the inner_width channels into (order + 1) groups of d_model along
    // dim 1: the first `order` are the gates x[...]; the last is v.
    auto splits = uc.split(/*split_size=*/d_model, /*dim=*/1);
    std::vector<torch::Tensor> x(splits.begin(), splits.end() - 1);  // order gates
    auto v = splits.back();

    // Implicit filter generation -- positional features.
    auto t = at::linspace(0.0, 1.0, l_filter, fopts).view({1, l_filter, 1});
    auto t_rescaled = at::linspace(0.0, static_cast<double>(l_filter - 1), l_filter, fopts)
                          .view({1, l_filter, 1});
    auto w = (2.0 * pi) * t_rescaled / static_cast<double>(l_filter);
    auto f = at::linspace(1e-4, static_cast<double>(bands - 1), bands, fopts)
                 .view({1, 1, bands});
    auto z = at::cat({t, at::cos(-f * w), at::sin(-f * w)}, /*dim=*/-1);

    // Filter MLP with sin(sin_freq * .) activations; final projection bias-free.
    auto h = at::linear(z, filter_linear1_weight, filter_linear1_bias);
    h = at::sin(sin_freq * h);
    h = at::linear(h, filter_linear2_weight, filter_linear2_bias);
    h = at::sin(sin_freq * h);
    h = at::linear(h, filter_linear3_weight, filter_linear3_bias);
    h = at::sin(sin_freq * h);
    h = at::linear(h, filter_linear_final_weight);  // no bias

    // Exponential modulation, then the additive filter bias.
    auto decay = at::exp(-t * exp_mod_deltas.abs());
    h = h * (decay + exp_mod_shift);
    h = h + filter_bias.view({1, 1, d_model});

    auto k = h.transpose(0, 1).reshape({1, d_model, l_filter});
    auto bias_reshaped = filter_bias.reshape({1, d_model});

    // Order-2 gating + FFT convolution. reversed(x[1:]) iterates the gates after
    // the first, newest-first; for order=2 this is the single gate x[1] at o=0.
    const int64_t fft_size = 2 * l_filter;
    for (size_t step = 0; step + 1 < x.size(); ++step) {
        const int64_t o = static_cast<int64_t>(step);
        const auto& x_i = x[x.size() - 1 - step];  // reversed(x[1:])

        v = v * x_i;

        // Forward rfft uses default norm (None); the filter spectrum is scaled
        // by 1/fft_size as a complex-tensor divide, matching the reference.
        auto k_o = k.select(/*dim=*/0, o);  // [d_model, l_filter]
        auto k_f = at::fft_rfft(k_o.to(torch::kFloat32), /*n=*/fft_size, /*dim=*/-1,
                                /*norm=*/c10::nullopt)
                       / static_cast<double>(fft_size);
        auto v_f = at::fft_rfft(v.to(torch::kFloat32), /*n=*/fft_size, /*dim=*/-1,
                                /*norm=*/c10::nullopt);

        // Frequency-domain multiply (k_f broadcasts over batch), inverse rfft
        // with norm='forward', then keep the causal head.
        auto y = at::fft_irfft(v_f * k_f, /*n=*/fft_size, /*dim=*/-1,
                               /*norm=*/c10::optional<c10::string_view>("forward"));
        y = y.narrow(/*dim=*/-1, /*start=*/0, /*length=*/l_filter);

        // Skip connection with the per-order bias (uses the gated v).
        auto bias_o = bias_reshaped.select(/*dim=*/0, o);  // [d_model]
        v = y + v * bias_o.unsqueeze(-1);
    }

    // Final gate with the first split, then back to [batch, seq, d_model].
    auto y = (v * x[0]).transpose(1, 2);

    // Pad the sequence axis back to seq_len when l_filter was clamped by l_max.
    if (l_filter < seq_len) {
        y = at::constant_pad_nd(
            y, /*pad=*/std::vector<int64_t>{0, 0, 0, seq_len - l_filter}, /*value=*/0);
    }

    // Output projection, then the first residual addition.
    auto hyena_out = at::linear(y, out_proj_weight, out_proj_bias);
    residual = hyena_out + residual;

    // Second LayerNorm.
    normed = layer_norm_fp32(residual, norm2_weight, norm2_bias, layer_norm_eps);

    // MLP with tanh-approximate GELU.
    auto mlp_out = at::linear(normed, mlp_fc1_weight, mlp_fc1_bias);
    mlp_out = at::gelu(mlp_out, /*approximate=*/"tanh");
    mlp_out = at::linear(mlp_out, mlp_fc2_weight, mlp_fc2_bias);

    // Final residual addition.
    auto result = mlp_out + residual;

    TORCH_CHECK(result.sizes() == output.sizes(),
                "computed output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Complete Hyena forward block via ATen / cuFFT / cuDNN / cuBLAS: "
          "LayerNorm, input projection, grouped short conv, sinusoidal "
          "implicit-filter MLP, order-2 FFT convolution with gating "
          "(rfft(n=2L)/N filter spectrum, irfft(norm='forward')[:L], per-order "
          "bias skip), output projection + residual, LayerNorm, tanh-GELU MLP "
          "+ residual. fp32 throughout (complex64 FFT intermediates); "
          "destination-passing style.");
}
