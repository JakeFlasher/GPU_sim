#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// AltUp hidden-state collapse with magnitude normalization.
//
// Reference (problem_052_altup_hidden_state_collapse_with_magnitude_normalization):
//     first_stream = hidden_states[0]                              # [B, S, H] bf16
//     target_magnitude = sqrt(mean(first_stream.f32 ** 2, -1, keepdim))   # [B, S, 1] f32
//     collapsed = [first_stream.f32]
//     # stream 1
//     proj1 = matmul(hidden_states[1].f32, unembed_proj_1.f32.t())        # [B, S, H] f32
//     cur1  = sqrt(maximum(mean(proj1 ** 2, -1, keepdim), epsilon))       # [B, S, 1] f32
//     collapsed.append(proj1 * (target_magnitude / cur1))
//     # stream 2
//     proj2 = matmul(hidden_states[2].f32, unembed_proj_2.f32.t())        # [B, S, H] f32
//     cur2  = sqrt(maximum(mean(proj2 ** 2, -1, keepdim), epsilon))       # [B, S, 1] f32
//     collapsed.append(proj2 * (target_magnitude / cur2))
//     output = mean(stack(collapsed, dim=0), dim=0).to(bf16)             # [B, S, H] bf16
//
// Per definition.json (insertion order):
//   hidden_states  : [altup_num_inputs=3, batch_size, seq_len, hidden_size=3072]  bf16
//   unembed_proj_1 : [hidden_size, hidden_size]                                   bf16
//   unembed_proj_2 : [hidden_size, hidden_size]                                   bf16
//   epsilon        : scalar                                                       f32
//   output         : [batch_size, seq_len, hidden_size]                           bf16 (DPS, pre-alloc)
//
// epsilon is a scalar (shape == null in definition.json) so it binds as a C++
// double, NOT a tensor.
//
// Precision contract (CLAUDE.md Rule 8): the reference casts every stream and
// projection weight to float32, runs the matmuls and magnitude reductions in
// float32, and only casts the final averaged result back to bfloat16. This port
// reproduces those cast points exactly: all intermediate math is float32; the
// single bf16 cast happens on the final mean. The unembed weights are cast to
// float32 and THEN transposed (.to(f32).t()), matching the reference order.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// two float32 matmuls route through the same cuBLAS path the reference's
// torch.matmul uses. Output is written in place into the pre-allocated
// destination (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& unembed_proj_1,
    const torch::Tensor& unembed_proj_2,
    double epsilon,
    torch::Tensor output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(unembed_proj_1, "unembed_proj_1");
    check_bf16_cuda(unembed_proj_2, "unembed_proj_2");
    check_bf16_cuda(output, "output");
    TORCH_CHECK(hidden_states.dim() == 4,
                "hidden_states must be 4D [altup_num_inputs, B, S, H], got ",
                hidden_states.dim(), "D");
    TORCH_CHECK(hidden_states.size(0) >= 3,
                "hidden_states must have at least 3 AltUp streams, got ",
                hidden_states.size(0));
    TORCH_CHECK(unembed_proj_1.dim() == 2, "unembed_proj_1 must be 2D [H, H]");
    TORCH_CHECK(unembed_proj_2.dim() == 2, "unembed_proj_2 must be 2D [H, H]");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const auto f32 = torch::kFloat32;
    const auto eps = static_cast<float>(epsilon);

    // first_stream = hidden_states[0]; collapsed[0] = first_stream.to(f32)
    auto first_stream = hidden_states.select(0, 0);            // [B, S, H] bf16 (view)
    auto first_stream_f32 = first_stream.to(f32);             // [B, S, H] f32

    // target_magnitude = sqrt(mean(first_stream.f32 ** 2, -1, keepdim)) -> [B, S, 1] f32
    auto target_magnitude = at::sqrt(
        first_stream_f32.pow(2).mean(-1, /*keepdim=*/true));

    // unembed weights: cast to f32 FIRST, then transpose (matches .to(f32).t()).
    auto w1_t = unembed_proj_1.to(f32).t();                    // [H, H] f32 (view)
    auto w2_t = unembed_proj_2.to(f32).t();                    // [H, H] f32 (view)

    // stream 1
    auto proj1 = at::matmul(hidden_states.select(0, 1).to(f32), w1_t);  // [B, S, H] f32
    auto cur1 = at::sqrt(at::clamp_min(
        proj1.pow(2).mean(-1, /*keepdim=*/true), eps));        // maximum(mean, eps)
    auto norm1 = proj1 * (target_magnitude / cur1);            // broadcast [B,S,1]

    // stream 2
    auto proj2 = at::matmul(hidden_states.select(0, 2).to(f32), w2_t);  // [B, S, H] f32
    auto cur2 = at::sqrt(at::clamp_min(
        proj2.pow(2).mean(-1, /*keepdim=*/true), eps));
    auto norm2 = proj2 * (target_magnitude / cur2);

    // output = mean(stack([first_stream_f32, norm1, norm2], 0), 0).to(bf16)
    auto stacked = at::stack({first_stream_f32, norm1, norm2}, /*dim=*/0);  // [3, B, S, H] f32
    auto result = stacked.mean(/*dim=*/0).to(torch::kBFloat16);             // [B, S, H] bf16

    TORCH_CHECK(result.sizes() == output.sizes(),
                "computed output shape ", result.sizes(),
                " does not match pre-allocated output ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "AltUp hidden-state collapse with magnitude normalization. Pure "
          "ATen/cuBLAS dispatch: two float32 matmuls (hidden_states[1/2] @ "
          "unembed_proj_{1,2}.t()), float32 magnitude normalization to the "
          "first stream's RMS magnitude (clamped at epsilon), then a 3-stream "
          "mean cast back to bf16. Destination-passing style.");
}
