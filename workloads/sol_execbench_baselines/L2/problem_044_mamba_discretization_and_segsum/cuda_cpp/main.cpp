#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <string>
#include <vector>

// Mamba SSM parameter discretization plus segment-sum, producing the decay /
// attention-like matrices used by the chunk-parallel scan.
//
// Reference (problem_044_mamba_discretization_and_segsum): a flat tensor-algebra
// chain with no inter-chunk recurrence -- every step is a single ATen op, so the
// whole port is a straight-line op-for-op transcription with no host loop.
//
// Dtype profile follows the reference exactly. All inputs arrive bf16; A_log,
// dt, B and C are promoted to fp32 for the numerically-sensitive discretization
// and contractions. L / G / M are cast back to bf16 on the way out while
// A_cumsum and decay_states stay fp32, matching the definition.json output
// dtypes.
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states : [batch, num_chunks, chunk_size, num_heads, head_dim]  bf16
//     dt            : [batch, num_chunks, chunk_size, num_heads]            bf16
//     A_log         : [num_heads]                                          bf16
//     B             : [batch, num_chunks, chunk_size, n_groups, d_state]    bf16
//     C             : [batch, num_chunks, chunk_size, n_groups, d_state]    bf16
//   outputs (DPS pre-alloc, written in place):
//     L            : [batch, num_heads, num_chunks, chunk_size, chunk_size] bf16
//     G            : [batch, num_chunks, chunk_size, chunk_size, num_heads] bf16
//     M            : [batch, num_chunks, chunk_size, chunk_size, num_heads] bf16
//     A_cumsum     : [batch, num_heads, num_chunks, chunk_size]            fp32
//     decay_states : [batch, num_heads, num_chunks, chunk_size]            fp32
//
// Implementation: pure ATen dispatch -- NO custom CUDA kernels. The einsums
// route through the same contraction backends torch uses; cumsum / tril /
// masked_fill / softplus / exp mirror the reference one for one.

// Stable segment sum: lower-triangular cumulative-sum matrix from a [..., T]
// vector. Expands a trailing T axis, zeroes the strictly-upper part, cumulates
// along the second-to-last axis, then masks the upper triangle (above the main
// diagonal) to -inf. Transcribed directly from the reference helper.
static at::Tensor segment_sum(const at::Tensor& input_tensor) {
    const int64_t chunk_size = input_tensor.size(-1);
    const auto device = input_tensor.device();

    // Expand to [..., chunk_size, chunk_size] with a fresh trailing axis.
    auto sizes = input_tensor.sizes().vec();
    sizes.push_back(chunk_size);
    auto input_expanded = input_tensor.unsqueeze(-1).expand(sizes);

    // Lower-triangular mask excluding the diagonal (diagonal = -1).
    auto ones = at::ones({chunk_size, chunk_size},
                         at::TensorOptions().dtype(at::kBool).device(device));
    auto mask = at::tril(ones, /*diagonal=*/-1);
    auto input_masked = input_expanded.masked_fill(~mask, 0);

    // Cumulative sum along the second-to-last dimension.
    auto tensor_segsum = at::cumsum(input_masked, /*dim=*/-2);

    // Keep the lower triangle including the diagonal; everything above -> -inf.
    auto mask_final = at::tril(ones, /*diagonal=*/0);
    tensor_segsum = tensor_segsum.masked_fill(
        ~mask_final, -std::numeric_limits<float>::infinity());

    return tensor_segsum;
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& dt,
    const torch::Tensor& A_log,
    const torch::Tensor& B,
    const torch::Tensor& C,
    torch::Tensor L,
    torch::Tensor G,
    torch::Tensor M,
    torch::Tensor A_cumsum,
    torch::Tensor decay_states
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
    TORCH_CHECK(dt.is_cuda(), "dt must be a CUDA tensor");
    TORCH_CHECK(A_log.is_cuda(), "A_log must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(hidden_states.dim() == 5,
                "hidden_states must be 5D [b, nc, cs, h, d], got ",
                hidden_states.dim(), "D");
    TORCH_CHECK(B.dim() == 5, "B must be 5D [b, nc, cs, g, s], got ", B.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t num_heads = hidden_states.size(3);
    const int64_t n_groups = B.size(3);

    // Discretize A: log-scale -> continuous-time decay, fp32.
    auto A = -at::exp(A_log.to(at::kFloat));                 // [num_heads]

    // Softplus on dt for numerical stability, then discretize against A.
    auto dt_soft = at::softplus(dt.to(at::kFloat));          // [b, nc, cs, h]
    auto dt_permuted = dt_soft.permute({0, 3, 1, 2});        // [b, h, nc, cs]
    auto A_discrete =
        A.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * dt_permuted;  // [b, h, nc, cs]

    // Cumulative decay factors.
    auto A_cumsum_result = at::cumsum(A_discrete, /*dim=*/-1);     // [b, h, nc, cs]

    // Intra-chunk decay matrix L = exp(segment_sum(A_discrete)).
    auto L_result = at::exp(segment_sum(A_discrete));  // [b, h, nc, cs, cs]

    // Broadcast B / C from n_groups up to num_heads via repeat_interleave.
    const int64_t repeats = num_heads / n_groups;
    auto B_expanded = at::repeat_interleave(B, repeats, /*dim=*/3);
    auto C_expanded = at::repeat_interleave(C, repeats, /*dim=*/3);

    // G[b,nc,i,j,h] = sum_d C[b,nc,i,h,d] * B[b,nc,j,h,d], in fp32.
    auto G_result = at::einsum(
        "bcihd,bcjhd->bcijh",
        {C_expanded.to(at::kFloat), B_expanded.to(at::kFloat)});  // [b, nc, cs, cs, h]

    // M = G * L (align L into G's axis order first).
    auto L_permuted = L_result.permute({0, 2, 3, 4, 1});  // [b, nc, cs, cs, h]
    auto M_result = G_result * L_permuted;                // [b, nc, cs, cs, h]

    // State-update decay factors: exp(A_cumsum[...,-1:] - A_cumsum).
    auto A_last = A_cumsum_result.slice(/*dim=*/-1,
                                        /*start=*/A_cumsum_result.size(-1) - 1,
                                        /*end=*/A_cumsum_result.size(-1));  // keepdim
    auto decay_states_result = at::exp(A_last - A_cumsum_result);  // [b, h, nc, cs]

    // Destination-passing-style writes (definition.json output order). L / G / M
    // cast back to bf16; A_cumsum / decay_states remain fp32.
    auto L_out = L_result.to(at::kBFloat16);
    auto G_out = G_result.to(at::kBFloat16);
    auto M_out = M_result.to(at::kBFloat16);

    TORCH_CHECK(L_out.sizes() == L.sizes(),
                "L shape ", L_out.sizes(), " does not match pre-allocated ", L.sizes());
    TORCH_CHECK(G_out.sizes() == G.sizes(),
                "G shape ", G_out.sizes(), " does not match pre-allocated ", G.sizes());
    TORCH_CHECK(M_out.sizes() == M.sizes(),
                "M shape ", M_out.sizes(), " does not match pre-allocated ", M.sizes());
    TORCH_CHECK(A_cumsum_result.sizes() == A_cumsum.sizes(),
                "A_cumsum shape ", A_cumsum_result.sizes(),
                " does not match pre-allocated ", A_cumsum.sizes());
    TORCH_CHECK(decay_states_result.sizes() == decay_states.sizes(),
                "decay_states shape ", decay_states_result.sizes(),
                " does not match pre-allocated ", decay_states.sizes());

    L.copy_(L_out);
    G.copy_(G_out);
    M.copy_(M_out);
    A_cumsum.copy_(A_cumsum_result);
    decay_states.copy_(decay_states_result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Mamba SSM discretization plus segment-sum decay / attention matrices. "
        "Pure ATen dispatch: softplus / exp discretization, cumsum decay, the "
        "tril-masked segment-sum helper, the C/B and L contractions reproduced "
        "op-for-op from the reference. bf16 inputs promoted to fp32 for the "
        "sensitive math; L / G / M returned bf16, A_cumsum / decay_states fp32. "
        "Destination-passing style (5 outputs).");
}
