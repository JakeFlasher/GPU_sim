#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

// Audio encoder sinusoidal position embedding with 3-stage Conv2d downsampling.
//
// Reference (problem_015_audio_sinusoidal_position_embedding_with_conv_projection),
// op-for-op:
//   x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
//   x = F.gelu(x)
//   x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
//   x = F.gelu(x)
//   x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
//   x = F.gelu(x)
//   b, c, f, t = x.size()
//   x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
//   x = F.linear(x, conv_out_weight)                 # no bias
//   x = x * embed_scale
//   seq_len = x.shape[1]
//   pos_embed = positional_embedding[:seq_len, :].unsqueeze(0)
//   x = x + pos_embed
//
// Per definition.json (insertion order):
//   input_features       : [batch_size, one=1, num_mel_bins=80, time_dim]  bf16
//   conv2d1_weight       : [downsample_hidden_size=384, 1, 3, 3]           bf16
//   conv2d1_bias         : [384]                                           bf16
//   conv2d2_weight       : [384, 384, 3, 3]                                bf16
//   conv2d2_bias         : [384]                                           bf16
//   conv2d3_weight       : [384, 384, 3, 3]                                bf16
//   conv2d3_bias         : [384]                                           bf16
//   conv_out_weight      : [d_model=1024, conv_out_dim=3840]               bf16
//   positional_embedding : [max_source_positions=1500, d_model=1024]       bf16
//   embed_scale          : scalar                                          fp32
//   hidden_states        : [batch_size, time_after_conv, d_model]          bf16  (DPS)
//
// All three convolutions use stride 2 and padding 1, baked into the reference
// run() (the workload table describes only this fixed configuration). The GELU
// activations use the default exact-erf variant -- F.gelu() with no
// approximate= argument. The final projection is bias-free, so
// at::linear(x, conv_out_weight) reproduces x @ conv_out_weight.t() exactly.
//
// Precision: everything stays bfloat16 to match the reference, which never
// casts up. embed_scale arrives as an fp32 scalar and is folded in as a plain
// scalar multiply (the tensor remains bf16). Wider accumulation inside the
// conv / matmul kernels is left to the ATen / cuDNN / cuBLAS dispatch, which
// already accumulates in fp32 for bf16 inputs.
//
// Implementation: pure ATen / cuDNN / cuBLAS dispatch -- no custom CUDA
// kernels. Output is written in place into the pre-allocated destination
// (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& input_features,
    const torch::Tensor& conv2d1_weight,
    const torch::Tensor& conv2d1_bias,
    const torch::Tensor& conv2d2_weight,
    const torch::Tensor& conv2d2_bias,
    const torch::Tensor& conv2d3_weight,
    const torch::Tensor& conv2d3_bias,
    const torch::Tensor& conv_out_weight,
    const torch::Tensor& positional_embedding,
    double embed_scale,
    torch::Tensor hidden_states
) {
    check_bf16_cuda(input_features, "input_features");
    check_bf16_cuda(conv2d1_weight, "conv2d1_weight");
    check_bf16_cuda(conv2d1_bias, "conv2d1_bias");
    check_bf16_cuda(conv2d2_weight, "conv2d2_weight");
    check_bf16_cuda(conv2d2_bias, "conv2d2_bias");
    check_bf16_cuda(conv2d3_weight, "conv2d3_weight");
    check_bf16_cuda(conv2d3_bias, "conv2d3_bias");
    check_bf16_cuda(conv_out_weight, "conv_out_weight");
    check_bf16_cuda(positional_embedding, "positional_embedding");
    check_bf16_cuda(hidden_states, "hidden_states");
    TORCH_CHECK(input_features.dim() == 4,
                "input_features must be 4D [N, 1, mel, time], got ", input_features.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(input_features.device());

    const std::vector<int64_t> stride = {2, 2};
    const std::vector<int64_t> padding = {1, 1};

    // Three stride-2 / pad-1 convolutions, each immediately followed by GELU.
    auto x = at::conv2d(input_features, conv2d1_weight, conv2d1_bias, stride, padding);
    x = at::gelu(x);
    x = at::conv2d(x, conv2d2_weight, conv2d2_bias, stride, padding);
    x = at::gelu(x);
    x = at::conv2d(x, conv2d3_weight, conv2d3_bias, stride, padding);
    x = at::gelu(x);

    // (batch, channels, freq, time) -> (batch, time, channels * freq).
    const int64_t b = x.size(0);
    const int64_t c = x.size(1);
    const int64_t f = x.size(2);
    const int64_t t = x.size(3);
    x = x.permute({0, 3, 1, 2}).contiguous().view({b, t, c * f});

    // Bias-free linear projection to d_model.
    x = at::linear(x, conv_out_weight);

    // Scale, then add the sliced, broadcast positional embeddings.
    x = x * embed_scale;
    const int64_t seq_len = x.size(1);
    auto pos_embed = positional_embedding.narrow(/*dim=*/0, /*start=*/0, /*length=*/seq_len).unsqueeze(0);
    x = x + pos_embed;

    TORCH_CHECK(x.sizes() == hidden_states.sizes(),
                "computed output shape ", x.sizes(),
                " does not match pre-allocated ", hidden_states.sizes());
    hidden_states.copy_(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Audio sinusoidal position embedding with 3-stage stride-2 Conv2d "
          "downsampling. Pure ATen / cuDNN / cuBLAS dispatch: conv2d + exact "
          "GELU x3, permute/reshape to (batch, time, channels*freq), bias-free "
          "at::linear projection to d_model, embed-scale multiply, add of the "
          "sliced broadcast positional embedding. bfloat16; destination-passing "
          "style.");
}
