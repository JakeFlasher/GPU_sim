#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <string>
#include <vector>

// Mamba-2 chunk-based parallel scan with segment sum.
//
// Reference (problem_043_mamba_chunk_scan_with_segsum): the sequence is padded
// to a whole number of chunks, intra-chunk attention-like contributions are
// formed from a tril-masked segment-sum decay matrix, per-chunk states are
// summarized, the states are propagated across chunks by an exponential-decay
// recurrence, and the diagonal (intra) and off-diagonal (inter) contributions
// are recombined and the D skip connection added.
//
// The inter-chunk propagation in the reference is a single einsum against a
// lower-triangular decay_chunk matrix. Because decay_chunk factors
// multiplicatively along the chunk axis, that einsum is exactly the linear
// recurrence new_states[c] = exp(chunk_end[c]) * new_states[c-1] + state[c],
// seeded with the initial state. num_chunks is a static shape (seq_len /
// chunk_size after padding), so the propagation is expressed below as a host
// for-loop carrying the running state -- a legal host loop, no custom kernel.
//
// Dtype profile follows the reference exactly. All inputs arrive bf16 and are
// promoted to fp32 for the whole scan; both outputs are cast back to bf16 at
// the end, matching the definition.json output dtypes.
//
// Per definition.json (insertion order):
//   inputs:
//     hidden_states  : [batch, seq_len, num_heads, head_dim]            bf16
//     A              : [batch, num_heads, seq_len]                      bf16
//     B              : [batch, seq_len, n_groups, state_size]           bf16
//     C              : [batch, seq_len, n_groups, state_size]           bf16
//     D              : [num_heads]                                      bf16
//     initial_states : [batch, num_heads, head_dim, state_size]         bf16
//   outputs (DPS pre-alloc, written in place):
//     output         : [batch, seq_len, num_heads * head_dim]           bf16
//     final_state    : [batch, num_heads, head_dim, state_size]         bf16
//
// Implementation: pure ATen dispatch -- NO custom CUDA kernels. The einsums
// route through the same contraction backends torch uses; cumsum / tril /
// masked_fill / pad / exp and the chunk recurrence mirror the reference.

// Stable segment sum: lower-triangular cumulative-sum matrix from a [..., T]
// vector, with the strictly-upper triangle set to -inf. Transcribed from the
// reference helper.
static at::Tensor segment_sum(const at::Tensor& input_tensor) {
    const int64_t chunk_size = input_tensor.size(-1);
    const auto device = input_tensor.device();

    auto sizes = input_tensor.sizes().vec();
    sizes.push_back(chunk_size);
    auto expanded = input_tensor.unsqueeze(-1).expand(sizes);

    auto ones = at::ones({chunk_size, chunk_size},
                         at::TensorOptions().dtype(at::kBool).device(device));
    auto mask = at::tril(ones, /*diagonal=*/-1);
    auto masked = expanded.masked_fill(~mask, 0);

    auto tensor_segsum = at::cumsum(masked, /*dim=*/-2);

    auto mask_final = at::tril(ones, /*diagonal=*/0);
    tensor_segsum = tensor_segsum.masked_fill(
        ~mask_final, -std::numeric_limits<float>::infinity());

    return tensor_segsum;
}

// Pad along the seq_len dimension with pad_size zero rows. Mirrors the
// reference helper: rank-4 tensors pad dim 1, rank-3 tensors pad dim 1 as well
// (third-from-last in F.pad's trailing-first ordering).
static at::Tensor pad_seq(const at::Tensor& t, int64_t pad_size) {
    if (pad_size == 0) {
        return t;
    }
    if (t.dim() == 4) {
        return at::constant_pad_nd(t, {0, 0, 0, 0, 0, pad_size, 0, 0}, 0);
    }
    return at::constant_pad_nd(t, {0, 0, 0, pad_size, 0, 0}, 0);
}

// Reshape into chunks after padding the seq_len axis.
static at::Tensor reshape_into_chunks(const at::Tensor& t, int64_t pad_size,
                                      int64_t chunk_size) {
    auto padded = pad_seq(t, pad_size);
    if (padded.dim() == 3) {
        return padded.reshape({padded.size(0), -1, chunk_size, padded.size(2)});
    }
    return padded.reshape(
        {padded.size(0), -1, chunk_size, padded.size(2), padded.size(3)});
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& D,
    const torch::Tensor& initial_states,
    torch::Tensor output,
    torch::Tensor final_state
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(D.is_cuda(), "D must be a CUDA tensor");
    TORCH_CHECK(initial_states.is_cuda(), "initial_states must be a CUDA tensor");
    TORCH_CHECK(hidden_states.dim() == 4,
                "hidden_states must be 4D [b, s, h, d], got ",
                hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t num_heads = hidden_states.size(2);
    const int64_t head_dim = hidden_states.size(3);
    const int64_t state_size = 256;
    const int64_t chunk_size = 256;

    const int64_t pad_size = (chunk_size - seq_len % chunk_size) % chunk_size;

    // Promote everything to fp32 for the scan.
    auto hidden_states_f = hidden_states.to(at::kFloat);
    auto A_f = A.to(at::kFloat);
    auto B_f = B.to(at::kFloat);
    auto C_f = C.to(at::kFloat);
    auto D_f = D.to(at::kFloat);
    auto initial_states_f = initial_states.to(at::kFloat);

    // Broadcast B / C from n_groups=1 up to num_heads.
    auto B_expanded = B_f.expand({batch_size, seq_len, num_heads, state_size});
    auto C_expanded = C_f.expand({batch_size, seq_len, num_heads, state_size});

    // D skip connection (computed before chunking, on the padded sequence).
    auto hidden_states_padded = pad_seq(hidden_states_f, pad_size);
    auto D_residual =
        D_f.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * hidden_states_padded;

    // Reshape inputs into chunks.
    auto hidden_states_chunked =
        reshape_into_chunks(hidden_states_f, pad_size, chunk_size);
    auto A_transposed = A_f.transpose(1, 2);  // [b, s, h]
    auto A_chunked = reshape_into_chunks(A_transposed, pad_size, chunk_size);
    auto B_chunked = reshape_into_chunks(B_expanded, pad_size, chunk_size);
    auto C_chunked = reshape_into_chunks(C_expanded, pad_size, chunk_size);

    const int64_t num_chunks = A_chunked.size(1);

    // Cumulative decay within each chunk.
    auto A_chunked_perm = A_chunked.permute({0, 3, 1, 2});  // [b, h, nc, cs]
    auto A_cumsum = at::cumsum(A_chunked_perm, /*dim=*/-1);

    // Intra-chunk (diagonal) output.
    auto L = at::exp(segment_sum(A_chunked_perm));  // [b, h, nc, cs, cs]
    auto G = at::einsum("bcihs,bcjhs->bcijh", {C_chunked, B_chunked});
    auto L_perm = L.permute({0, 2, 3, 4, 1});       // [b, nc, cs, cs, h]
    auto M = G * L_perm;
    auto Y_diag = at::einsum("bcijh,bcjhd->bcihd", {M, hidden_states_chunked});

    // Per-chunk states (right factor).
    auto A_last = A_cumsum.slice(/*dim=*/-1, /*start=*/A_cumsum.size(-1) - 1,
                                 /*end=*/A_cumsum.size(-1));
    auto decay_states = at::exp(A_last - A_cumsum);          // [b, h, nc, cs]
    auto decay_states_perm = decay_states.permute({0, 2, 3, 1});  // [b, nc, cs, h]
    auto B_decay = B_chunked * decay_states_perm.unsqueeze(-1);
    auto states =
        at::einsum("bcths,bcthd->bchds", {B_decay, hidden_states_chunked});
    // states: [b, nc, h, d, s]

    // Inter-chunk recurrence (middle factor). The reference einsum against the
    // lower-triangular decay_chunk matrix is the linear scan
    //   new_states[c+1] = exp(chunk_end[c]) * new_states[c] + states[c],
    // seeded with the initial state. chunk_end[c] = A_cumsum[..., c, -1].
    auto A_chunk_ends = A_cumsum.select(/*dim=*/-1, /*index=*/-1);  // [b, h, nc]
    auto chunk_end_decay = at::exp(A_chunk_ends);                   // [b, h, nc]

    // states_out collects new_states[0 .. nc-1]; final_state is new_states[nc].
    std::vector<at::Tensor> states_out_list;
    states_out_list.reserve(num_chunks);
    auto running = initial_states_f;  // new_states[0] = initial state, [b, h, d, s]
    for (int64_t c = 0; c < num_chunks; ++c) {
        states_out_list.push_back(running);
        // decay factor for this step broadcasts over (head_dim, state_size).
        auto decay_c = chunk_end_decay.select(/*dim=*/-1, /*index=*/c)
                           .unsqueeze(-1)
                           .unsqueeze(-1);          // [b, h, 1, 1]
        auto state_c = states.select(/*dim=*/1, /*index=*/c);  // [b, h, d, s]
        running = decay_c * running + state_c;
    }
    auto states_out = at::stack(states_out_list, /*dim=*/1);  // [b, nc, h, d, s]
    auto final_state_result = running;                        // [b, h, d, s]

    // State -> output conversion (left factor).
    auto state_decay_out = at::exp(A_cumsum);                       // [b, h, nc, cs]
    auto state_decay_out_perm = state_decay_out.permute({0, 2, 3, 1});  // [b, nc, cs, h]
    auto C_times_states =
        at::einsum("bcths,bchds->bcthd", {C_chunked, states_out});
    auto Y_off = C_times_states * state_decay_out_perm.unsqueeze(-1);

    // Combine diagonal and off-diagonal contributions, fold back to seq_len,
    // add the D residual, drop the padding, flatten the head axes.
    auto y = Y_diag + Y_off;                                     // [b, nc, cs, h, d]
    y = y.reshape({batch_size, -1, num_heads, head_dim});       // [b, s_pad, h, d]
    y = y + D_residual;
    if (pad_size > 0) {
        y = y.slice(/*dim=*/1, /*start=*/0, /*end=*/seq_len);
    }
    auto output_result =
        y.reshape({batch_size, seq_len, num_heads * head_dim}).to(at::kBFloat16);
    auto final_state_out = final_state_result.to(at::kBFloat16);

    TORCH_CHECK(output_result.sizes() == output.sizes(),
                "output shape ", output_result.sizes(),
                " does not match pre-allocated ", output.sizes());
    TORCH_CHECK(final_state_out.sizes() == final_state.sizes(),
                "final_state shape ", final_state_out.sizes(),
                " does not match pre-allocated ", final_state.sizes());

    output.copy_(output_result);
    final_state.copy_(final_state_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Mamba-2 chunk-based parallel scan with segment sum. Pure ATen "
        "dispatch: pad / reshape into chunks, cumsum decay, the tril-masked "
        "segment-sum helper, the C/B and value contractions, and the "
        "inter-chunk state propagation expressed as a host for-loop carrying "
        "the running state (the lower-triangular decay einsum rewritten as the "
        "equivalent linear recurrence). bf16 inputs promoted to fp32 for the "
        "scan; output and final_state returned bf16. Destination-passing style "
        "(2 outputs).");
}
