#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>

// SD3.5 joint-attention input preparation, reproduced op-for-op from the
// reference run(). Three embedding pathways are combined: spatial image patches
// (patch-embed conv + learned positional encoding), temporal conditioning
// (sinusoidal timestep features plus a pooled-projection branch, summed), and
// projected text context. The kernel emits all three pathway outputs.
//
// Reference, op-for-op:
//   patch_embedded = F.conv2d(hidden_states, proj_weight, proj_bias, stride=2)
//   patch_embedded = patch_embedded.flatten(2).transpose(1, 2)   # [B, h*w, inner]
//   num_patches = patch_embedded.shape[1]
//   output_hidden_states = patch_embedded + pos_embed[:, :num_patches, :]
//
//   args = timestep.unsqueeze(-1) * freqs.unsqueeze(0)           # [B, half]
//   timestep_sinusoidal = cat([cos(args), sin(args)], dim=-1)    # [B, 2*half]
//   timestep_embed = linear(timestep_sinusoidal, t_lin1_w, t_lin1_b)
//   timestep_embed = silu(timestep_embed)
//   timestep_embed = linear(timestep_embed, t_lin2_w, t_lin2_b)
//
//   pooled_embed = linear(pooled_projections, p_lin1_w, p_lin1_b)
//   pooled_embed = silu(pooled_embed)
//   pooled_embed = linear(pooled_embed, p_lin2_w, p_lin2_b)
//
//   temb = timestep_embed + pooled_embed
//   output_encoder_hidden_states = linear(encoder_hidden_states, ctx_w, ctx_b)
//
// Per definition.json (input insertion order, then output order):
//   hidden_states            : [batch, in_channels=16, 128, 128]      fp32
//   encoder_hidden_states    : [batch, seq_len_context, 4096]         fp32
//   pooled_projections       : [batch, pooled_projection_dim=2048]    fp32
//   timestep                 : [batch]                                fp32 (tensor)
//   proj_weight              : [inner_dim=2432, 16, 2, 2]             fp32
//   proj_bias                : [2432]                                  fp32
//   pos_embed                : [1, num_patches, 2432]                 fp32
//   timestep_linear1_weight  : [2432, sinusoidal_dim=256]            fp32
//   timestep_linear1_bias    : [2432]                                 fp32
//   timestep_linear2_weight  : [2432, 2432]                           fp32
//   timestep_linear2_bias    : [2432]                                 fp32
//   pooled_linear1_weight    : [2432, 2048]                           fp32
//   pooled_linear1_bias      : [2432]                                 fp32
//   pooled_linear2_weight    : [2432, 2432]                           fp32
//   pooled_linear2_bias      : [2432]                                 fp32
//   context_embedder_weight  : [caption_projection_dim=2432, 4096]   fp32
//   context_embedder_bias    : [2432]                                 fp32
//   freqs                    : [sinusoidal_half_dim=128]             fp32
//   output_hidden_states         : [batch, num_patches, 2432]        fp32 DPS
//   temb                         : [batch, 2432]                      fp32 DPS
//   output_encoder_hidden_states : [batch, seq_len_context, 2432]    fp32 DPS
//
// timestep is a real per-batch tensor input, not a scalar; it is broadcast
// against freqs via the outer product timestep[:, None] * freqs[None, :].
//
// Implementation: pure ATen / cuDNN / cuBLAS dispatch -- NO custom CUDA kernels.
// at::conv2d uses the contractual stride 2 with no padding (the reference passes
// stride=2 only). at::linear reproduces each F.linear (x @ W^T + b), and
// at::silu / at::cos / at::sin / at::cat / at::narrow cover the rest. Everything
// stays fp32 throughout, matching the reference precision profile; wider
// accumulation inside the conv / matmul dispatch is left to ATen. The three
// outputs are written in place in definition order (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& encoder_hidden_states,
    const torch::Tensor& pooled_projections,
    const torch::Tensor& timestep,
    const torch::Tensor& proj_weight,
    const torch::Tensor& proj_bias,
    const torch::Tensor& pos_embed,
    const torch::Tensor& timestep_linear1_weight,
    const torch::Tensor& timestep_linear1_bias,
    const torch::Tensor& timestep_linear2_weight,
    const torch::Tensor& timestep_linear2_bias,
    const torch::Tensor& pooled_linear1_weight,
    const torch::Tensor& pooled_linear1_bias,
    const torch::Tensor& pooled_linear2_weight,
    const torch::Tensor& pooled_linear2_bias,
    const torch::Tensor& context_embedder_weight,
    const torch::Tensor& context_embedder_bias,
    const torch::Tensor& freqs,
    torch::Tensor output_hidden_states,
    torch::Tensor temb,
    torch::Tensor output_encoder_hidden_states
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    check_fp32_cuda(encoder_hidden_states, "encoder_hidden_states");
    check_fp32_cuda(pooled_projections, "pooled_projections");
    check_fp32_cuda(timestep, "timestep");
    check_fp32_cuda(proj_weight, "proj_weight");
    check_fp32_cuda(proj_bias, "proj_bias");
    check_fp32_cuda(pos_embed, "pos_embed");
    check_fp32_cuda(timestep_linear1_weight, "timestep_linear1_weight");
    check_fp32_cuda(timestep_linear1_bias, "timestep_linear1_bias");
    check_fp32_cuda(timestep_linear2_weight, "timestep_linear2_weight");
    check_fp32_cuda(timestep_linear2_bias, "timestep_linear2_bias");
    check_fp32_cuda(pooled_linear1_weight, "pooled_linear1_weight");
    check_fp32_cuda(pooled_linear1_bias, "pooled_linear1_bias");
    check_fp32_cuda(pooled_linear2_weight, "pooled_linear2_weight");
    check_fp32_cuda(pooled_linear2_bias, "pooled_linear2_bias");
    check_fp32_cuda(context_embedder_weight, "context_embedder_weight");
    check_fp32_cuda(context_embedder_bias, "context_embedder_bias");
    check_fp32_cuda(freqs, "freqs");
    check_fp32_cuda(output_hidden_states, "output_hidden_states");
    check_fp32_cuda(temb, "temb");
    check_fp32_cuda(output_encoder_hidden_states, "output_encoder_hidden_states");

    TORCH_CHECK(hidden_states.dim() == 4, "hidden_states must be 4D [B, C, H, W], got ",
                hidden_states.dim(), "D");
    TORCH_CHECK(timestep.dim() == 1, "timestep must be 1D [B], got ", timestep.dim(), "D");
    TORCH_CHECK(freqs.dim() == 1, "freqs must be 1D [half_dim], got ", freqs.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    // Patch embedding: stride-2 conv with no padding, then flatten the spatial
    // grid into a sequence and move channels last.
    const std::vector<int64_t> stride{2, 2};
    auto patch_embedded = at::conv2d(hidden_states, proj_weight, proj_bias, stride);
    patch_embedded = patch_embedded.flatten(2).transpose(1, 2);  // [B, h*w, inner_dim]

    // Add the learned positional embeddings, sliced to the patch count.
    const int64_t num_patches = patch_embedded.size(1);
    auto pos_embed_slice = pos_embed.narrow(/*dim=*/1, /*start=*/0, /*length=*/num_patches);
    auto out_hidden = patch_embedded + pos_embed_slice;

    // Sinusoidal timestep features: outer product of the per-batch timestep with
    // the frequency bands, then concatenated cos / sin halves.
    auto args = timestep.unsqueeze(-1) * freqs.unsqueeze(0);  // [B, half_dim]
    auto timestep_sinusoidal = at::cat({at::cos(args), at::sin(args)}, /*dim=*/-1);  // [B, 2*half_dim]

    // Timestep MLP: linear -> silu -> linear.
    auto timestep_embed = at::linear(timestep_sinusoidal, timestep_linear1_weight, timestep_linear1_bias);
    timestep_embed = at::silu(timestep_embed);
    timestep_embed = at::linear(timestep_embed, timestep_linear2_weight, timestep_linear2_bias);

    // Pooled-projection MLP: linear -> silu -> linear.
    auto pooled_embed = at::linear(pooled_projections, pooled_linear1_weight, pooled_linear1_bias);
    pooled_embed = at::silu(pooled_embed);
    pooled_embed = at::linear(pooled_embed, pooled_linear2_weight, pooled_linear2_bias);

    // Temporal conditioning is the sum of the two branches.
    auto out_temb = timestep_embed + pooled_embed;

    // Context embedder projects the text embeddings to the caption dimension.
    auto out_encoder = at::linear(encoder_hidden_states, context_embedder_weight, context_embedder_bias);

    TORCH_CHECK(out_hidden.sizes() == output_hidden_states.sizes(),
                "output_hidden_states shape ", out_hidden.sizes(),
                " does not match pre-allocated ", output_hidden_states.sizes());
    TORCH_CHECK(out_temb.sizes() == temb.sizes(),
                "temb shape ", out_temb.sizes(),
                " does not match pre-allocated ", temb.sizes());
    TORCH_CHECK(out_encoder.sizes() == output_encoder_hidden_states.sizes(),
                "output_encoder_hidden_states shape ", out_encoder.sizes(),
                " does not match pre-allocated ", output_encoder_hidden_states.sizes());

    output_hidden_states.copy_(out_hidden);
    temb.copy_(out_temb);
    output_encoder_hidden_states.copy_(out_encoder);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "SD3.5 joint-attention input preparation via pure ATen dispatch: "
          "stride-2 patch-embed conv flattened to a sequence plus sliced "
          "learned positional embeddings; sinusoidal timestep features "
          "(cos/sin of timestep x freqs) through a linear-silu-linear MLP "
          "summed with a pooled-projection linear-silu-linear MLP to form the "
          "temporal embedding; and a context-embedder linear projection of the "
          "text embeddings. Three outputs written in place (output_hidden_states, "
          "temb, output_encoder_hidden_states). fp32; destination-passing style.");
}
