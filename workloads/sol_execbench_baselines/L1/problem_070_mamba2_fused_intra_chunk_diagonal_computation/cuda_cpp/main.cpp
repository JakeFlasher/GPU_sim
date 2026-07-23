#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <string>

// Fused intra-chunk diagonal computation for Mamba2 SSD.
//
// Reference (problem_070_mamba2_fused_intra_chunk_diagonal_computation),
// reproduced op-for-op. The diagonal block of the chunked parallel scan
// combines a segment-sum causal decay mask with a B / C state contraction and
// applies the result to the hidden states.
//
//   CHUNK_SIZE = 128 ; NUM_HEADS = 32 ; N_GROUPS = 8
//   hidden_states : [batch, num_chunks, chunk_size, num_heads, head_dim]   bf16
//   A_cumsum      : [batch, num_heads, num_chunks, chunk_size]             bf16
//   B, C          : [batch, num_chunks, chunk_size, n_groups, state_size]  bf16
//
// segment_sum + decay mask (computed in fp32):
//   A_expanded   = A_cumsum[..., None].expand(..., CHUNK_SIZE).to(fp32)
//                  # [batch, num_heads, num_chunks, chunk_size, chunk_size]
//   mask         = tril(ones(C,C, bool), diagonal=-1)    # strictly-below diagonal
//   A_masked     = A_expanded.masked_fill(~mask, 0)
//   A_cumsum_seg = cumsum(A_masked, dim=-2)              # along the source axis
//   mask_diag    = tril(ones(C,C, bool), diagonal=0)     # includes the diagonal
//   segment_sum_A= A_cumsum_seg.masked_fill(~mask_diag, -inf)
//   L            = exp(segment_sum_A)
//
// B / C contraction G (computed in fp32):
//   B_f32, C_f32 = B.to(fp32), C.to(fp32)
//   B_expanded   = B_f32.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)
//   C_expanded   = C_f32.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)
//                  # both -> [batch, num_chunks, chunk_size, num_heads, state_size]
//   G[b,c,i,j,h] = sum_n C_expanded[b,c,i,h,n] * B_expanded[b,c,j,h,n]
//                  # via the broadcast outer product (C_for_G * B_for_G).sum(-1)
//                  # -> [batch, num_chunks, chunk_size_i, chunk_size_j, num_heads]
//
// mask application + diagonal output:
//   L_permuted   = L.permute(0, 2, 3, 4, 1)   # -> [batch, num_chunks, ci, cj, num_heads]
//   M            = G * L_permuted
//   hidden_f32   = hidden_states.to(fp32)
//   Y_diag[b,c,i,h,d] = sum_j M[b,c,i,j,h] * hidden_f32[b,c,j,h,d]
//                  # (M[...,None] * hidden_f32[:, :, None]).sum(dim=3)
//                  # -> [batch, num_chunks, chunk_size, num_heads, head_dim]
//   return Y_diag.to(bf16)
//
// Per definition.json (insertion order):
//   hidden_states : [batch, num_chunks, chunk_size, num_heads, head_dim]   bf16
//   A_cumsum      : [batch, num_heads, num_chunks, chunk_size]             bf16
//   B             : [batch, num_chunks, chunk_size, n_groups, state_size]  bf16
//   C             : [batch, num_chunks, chunk_size, n_groups, state_size]  bf16
//   Y_diag        : [batch, num_chunks, chunk_size, num_heads, head_dim]   bf16 DPS
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// two contractions are expressed through at::einsum to keep the index
// bookkeeping identical to the reference's broadcast-multiply-and-sum; the
// segment-sum decay path stays in fp32 (matching the reference's .to(fp32)
// before masked_fill / cumsum / exp) and the inputs are upcast to fp32 before
// the contractions, exactly where the reference casts them. The fp32 result
// is cast back to bf16 only on the final write. Output is written in place
// into the pre-allocated destination (destination-passing style).

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& A_cumsum,
    const torch::Tensor& B,
    const torch::Tensor& C,
    torch::Tensor Y_diag
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(A_cumsum, "A_cumsum");
    check_bf16_cuda(B, "B");
    check_bf16_cuda(C, "C");
    check_bf16_cuda(Y_diag, "Y_diag");
    TORCH_CHECK(hidden_states.dim() == 5,
                "hidden_states must be 5D [batch, num_chunks, chunk_size, num_heads, head_dim], got ",
                hidden_states.dim(), "D");
    TORCH_CHECK(A_cumsum.dim() == 4,
                "A_cumsum must be 4D [batch, num_heads, num_chunks, chunk_size], got ",
                A_cumsum.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kChunkSize = 128;
    constexpr int64_t kNumHeads = 32;
    constexpr int64_t kNGroups = 8;
    constexpr int64_t kHeadsPerGroup = kNumHeads / kNGroups;

    const int64_t state_size = B.size(4);

    // --- segment_sum(A) + exponential decay mask L (fp32). ---
    // A_expanded: [batch, num_heads, num_chunks, chunk_size, chunk_size]
    auto a_sizes = A_cumsum.sizes().vec();
    a_sizes.push_back(kChunkSize);
    auto A_expanded = A_cumsum.unsqueeze(-1).expand(a_sizes).to(at::kFloat);

    const auto bool_opts =
        at::TensorOptions().dtype(at::kBool).device(A_cumsum.device());
    auto ones_cc = at::ones({kChunkSize, kChunkSize}, bool_opts);
    auto mask = at::tril(ones_cc, /*diagonal=*/-1);            // strictly below diagonal
    auto mask_with_diag = at::tril(ones_cc, /*diagonal=*/0);   // includes diagonal

    // Zero the upper-triangular (and diagonal) entries, then cumulative sum
    // along the source axis (dim=-2).
    auto A_masked = A_expanded.masked_fill(mask.logical_not(), 0);
    auto A_cumsum_seg = at::cumsum(A_masked, /*dim=*/-2);
    auto neg_inf = -std::numeric_limits<double>::infinity();
    auto segment_sum_A = A_cumsum_seg.masked_fill(mask_with_diag.logical_not(), neg_inf);
    auto L = at::exp(segment_sum_A);  // [batch, num_heads, num_chunks, ci, cj]

    // --- B / C contraction G (fp32). ---
    auto B_f32 = B.to(at::kFloat);
    auto C_f32 = C.to(at::kFloat);

    // Expand the group dimension out to num_heads (group h maps to heads
    // [h*heads_per_group, (h+1)*heads_per_group)).
    auto B_expanded = at::repeat_interleave(B_f32, kHeadsPerGroup, /*dim=*/3);
    auto C_expanded = at::repeat_interleave(C_f32, kHeadsPerGroup, /*dim=*/3);
    // both -> [batch, num_chunks, chunk_size, num_heads, state_size]

    // G[b,c,i,j,h] = sum_n C_expanded[b,c,i,h,n] * B_expanded[b,c,j,h,n]
    // -> [batch, num_chunks, chunk_size_i, chunk_size_j, num_heads]
    auto G = at::einsum("bcihn,bcjhn->bcijh", {C_expanded, B_expanded});

    // --- Apply L to G to form M. ---
    // L: [batch, num_heads, num_chunks, ci, cj] -> [batch, num_chunks, ci, cj, num_heads]
    auto L_permuted = L.permute({0, 2, 3, 4, 1});
    auto M = G * L_permuted;  // [batch, num_chunks, ci, cj, num_heads]

    // --- Y_diag = sum over the source position cj of M applied to hidden. ---
    auto hidden_f32 = hidden_states.to(at::kFloat);
    // Y_diag[b,c,i,h,d] = sum_j M[b,c,i,j,h] * hidden_f32[b,c,j,h,d]
    // -> [batch, num_chunks, chunk_size, num_heads, head_dim]
    auto Y_diag_f32 = at::einsum("bcijh,bcjhd->bcihd", {M, hidden_f32});
    auto out = Y_diag_f32.to(at::kBFloat16);

    (void)state_size;

    // --- Destination-passing-style write. ---
    TORCH_CHECK(out.sizes() == Y_diag.sizes(),
                "Y_diag shape ", out.sizes(),
                " does not match pre-allocated ", Y_diag.sizes());
    Y_diag.copy_(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Mamba2 SSD intra-chunk diagonal computation. Pure ATen / cuBLAS "
          "dispatch: fp32 segment-sum decay mask L (lower-triangular masked "
          "cumsum then exp), B / C state contraction G via einsum, mask "
          "application M = G * L, and the diagonal output Y_diag = sum over "
          "source positions of M applied to the hidden states. bf16 inputs / "
          "output with fp32 intermediate compute; destination-passing style.");
}
