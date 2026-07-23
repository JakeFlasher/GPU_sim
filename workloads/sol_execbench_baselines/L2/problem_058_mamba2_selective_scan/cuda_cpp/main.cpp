#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <limits>
#include <string>

// Mamba2 selective state space model core computation.
//
// Reference (problem_058_mamba2_selective_scan), reproduced op-for-op:
// input projection, causal depthwise convolution with SiLU, time-step
// discretization (softplus + clamp), chunked selective scan (intra-chunk
// diagonal block + inter-chunk recurrent state propagation + state-to-output
// off-diagonal block), gated RMS normalization, and an output projection.
//
// Pinned constants (definition.json axes):
//   hidden_size = 8192 ; num_heads = 256 ; head_dim = 64
//   intermediate_size = 16384 ; ssm_state_size = 256 ; conv_kernel_size = 4
//   n_groups = 8 ; chunk_size = 128
//   groups_time_state_size = n_groups * ssm_state_size = 2048
//   conv_dim = intermediate_size + 2 * groups_time_state_size = 20480
//
// Per definition.json (input insertion order, then the single output):
//   hidden_states   : [batch, seq, hidden_size]                 bf16
//   in_proj_weight  : [projection_size, hidden_size]            bf16
//   conv1d_weight   : [conv_dim, 1, conv_kernel_size]           bf16
//   conv1d_bias     : [conv_dim]                                bf16
//   dt_bias         : [num_heads]                               bf16
//   A_log           : [num_heads]                               bf16
//   D               : [num_heads]                               bf16
//   norm_weight     : [intermediate_size]                       bf16
//   out_proj_weight : [hidden_size, intermediate_size]          bf16
//   time_step_limit_min : scalar fp32
//   time_step_limit_max : scalar fp32
//   layer_norm_epsilon  : scalar fp32
//   output          : [batch, seq, hidden_size]                 bf16 (DPS)
//
// Dtype profile (Rule 8): bf16 I/O. The projection and convolution stay in
// bf16 (the reference never upcasts there). The selective-scan body matches
// the reference's explicit `.float()` casts: hidden_states_ssm / B / C / A /
// dt are computed in fp32, the segment-sum decay masks / einsums / inter-chunk
// recurrence all run in fp32, and the result is cast back to bf16 exactly where
// the reference calls `.to(dtype)` (after combining y + D_residual, and again
// after the gated norm). The gated-norm variance is fp32 per the reference's
// `.float().pow(2).mean(...)`.
//
// The inter-chunk recurrence is a fixed-length scan over num_chunks (a static
// shape derived from the padded sequence length). The reference expresses it as
// a single `decay_chunk @ states_with_prev` einsum over a lower-triangular
// inter-chunk decay matrix; that is a legal closed-form contraction and is kept
// as an at::einsum here. An explicit host `for (c = 0; c < num_chunks; ++c)`
// carrying the running `states` would reproduce the same numbers; the einsum
// form is the op-for-op transcription of the reference and is retained.
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. The
// einsum contractions keep the index bookkeeping identical to the reference.
// Output is written in place into the pre-allocated destination.

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& in_proj_weight,
    const torch::Tensor& conv1d_weight,
    const torch::Tensor& conv1d_bias,
    const torch::Tensor& dt_bias,
    const torch::Tensor& A_log,
    const torch::Tensor& D,
    const torch::Tensor& norm_weight,
    const torch::Tensor& out_proj_weight,
    double time_step_limit_min,
    double time_step_limit_max,
    double layer_norm_epsilon,
    torch::Tensor output
) {
    check_bf16_cuda(hidden_states, "hidden_states");
    check_bf16_cuda(in_proj_weight, "in_proj_weight");
    check_bf16_cuda(conv1d_weight, "conv1d_weight");
    check_bf16_cuda(conv1d_bias, "conv1d_bias");
    check_bf16_cuda(dt_bias, "dt_bias");
    check_bf16_cuda(A_log, "A_log");
    check_bf16_cuda(D, "D");
    check_bf16_cuda(norm_weight, "norm_weight");
    check_bf16_cuda(out_proj_weight, "out_proj_weight");
    check_bf16_cuda(output, "output");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [batch, seq, hidden], got ",
                hidden_states.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kHiddenSize = 8192;
    constexpr int64_t kNumHeads = 256;
    constexpr int64_t kHeadDim = 64;
    constexpr int64_t kIntermediateSize = 16384;
    constexpr int64_t kSsmStateSize = 256;
    constexpr int64_t kConvKernelSize = 4;
    constexpr int64_t kNGroups = 8;
    constexpr int64_t kChunkSize = 128;
    constexpr int64_t kGroupsTimeStateSize = kNGroups * kSsmStateSize;  // 2048
    constexpr int64_t kConvDim =
        kIntermediateSize + 2 * kGroupsTimeStateSize;  // 20480
    constexpr int64_t kHeadsPerGroup = kNumHeads / kNGroups;  // 32

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    TORCH_CHECK(hidden_states.size(2) == kHiddenSize,
                "hidden_size must be ", kHiddenSize, ", got ", hidden_states.size(2));

    // 1. Input projection: (B, L, H) @ in_proj_weight^T -> (B, L, projection_size).
    torch::Tensor projected = at::matmul(hidden_states, in_proj_weight.t());

    // Split projections. gate_start indexes from the front so that gate spans
    // intermediate_size, hidden_B_C spans conv_dim, and dt is the final
    // num_heads columns.
    const int64_t projection_size = projected.size(-1);
    const int64_t gate_start =
        projection_size - kIntermediateSize - kConvDim - kNumHeads;
    torch::Tensor gate =
        projected.narrow(-1, gate_start, kIntermediateSize);
    torch::Tensor hidden_B_C =
        projected.narrow(-1, gate_start + kIntermediateSize, kConvDim);
    torch::Tensor dt =
        projected.narrow(-1, projection_size - kNumHeads, kNumHeads);

    // 2. Causal convolution. Transpose to (B, conv_dim, L), depthwise conv1d
    //    with symmetric padding conv_kernel_size - 1, keep the first seq_len
    //    output positions, SiLU, transpose back.
    torch::Tensor hidden_B_C_t = hidden_B_C.transpose(1, 2);
    // Explicitly-typed int64 vectors so the IntArrayRef conv1d overload is the
    // unique match (a braced-init is ambiguous against the string_view padding
    // overload when bias is a raw Tensor).
    const std::vector<int64_t> stride{1};
    const std::vector<int64_t> padding{kConvKernelSize - 1};
    const std::vector<int64_t> dilation{1};
    torch::Tensor conv_out = at::conv1d(
        hidden_B_C_t,
        conv1d_weight,
        conv1d_bias,
        /*stride=*/stride,
        /*padding=*/padding,
        /*dilation=*/dilation,
        /*groups=*/kConvDim);
    conv_out = conv_out.narrow(-1, 0, seq_len);
    // silu(x) = x * sigmoid(x).
    hidden_B_C = (conv_out * at::sigmoid(conv_out)).transpose(1, 2);

    // Split into hidden_states_ssm, B, C along the channel dim.
    torch::Tensor hidden_states_ssm =
        hidden_B_C.narrow(-1, 0, kIntermediateSize);
    torch::Tensor B =
        hidden_B_C.narrow(-1, kIntermediateSize, kGroupsTimeStateSize);
    torch::Tensor C = hidden_B_C.narrow(
        -1, kIntermediateSize + kGroupsTimeStateSize, kGroupsTimeStateSize);

    // 3. Selective state space model with chunking.

    // Discretize time step: dt = clamp(softplus(dt + dt_bias), min, max).
    dt = at::softplus(dt + dt_bias);
    dt = at::clamp(dt, time_step_limit_min, time_step_limit_max);

    // Reshape for the SSM computation (fp32 per the reference's .float()).
    hidden_states_ssm =
        hidden_states_ssm.view({batch_size, seq_len, kNumHeads, kHeadDim})
            .to(at::kFloat);
    B = B.view({batch_size, seq_len, kNGroups, kSsmStateSize}).to(at::kFloat);
    C = C.view({batch_size, seq_len, kNGroups, kSsmStateSize}).to(at::kFloat);

    // Repeat B and C for all heads in each group: tile the group dim.
    B = B.repeat({1, 1, kHeadsPerGroup, 1});  // (batch, seq, num_heads, state)
    C = C.repeat({1, 1, kHeadsPerGroup, 1});  // (batch, seq, num_heads, state)

    // Pad sequence up to a chunk_size multiple.
    const int64_t pad_size = (kChunkSize - seq_len % kChunkSize) % kChunkSize;

    // pad_tensor_4d pads the second-from-front (seq) axis of a 4D tensor laid
    // out (batch, seq, heads, last). F.pad's tuple (0,0, 0,0, 0,pad, 0,0) pads
    // the last four dims; constant_pad_nd takes the same low/high pairs in
    // last-dim-first order.
    auto pad_tensor_4d = [&](const torch::Tensor& x) -> torch::Tensor {
        if (pad_size > 0) {
            const std::vector<int64_t> p{0, 0, 0, 0, 0, pad_size, 0, 0};
            return at::constant_pad_nd(x, p, 0);
        }
        return x;
    };
    auto pad_tensor_3d = [&](const torch::Tensor& x) -> torch::Tensor {
        if (pad_size > 0) {
            const std::vector<int64_t> p{0, 0, 0, pad_size, 0, 0};
            return at::constant_pad_nd(x, p, 0);
        }
        return x;
    };

    // D residual (skip connection): D.float()[..., None] * padded hidden.
    torch::Tensor D_residual =
        D.to(at::kFloat).unsqueeze(-1) * pad_tensor_4d(hidden_states_ssm);

    // Discretize x and A.
    hidden_states_ssm = hidden_states_ssm * dt.unsqueeze(-1);
    torch::Tensor A = -at::exp(A_log.to(at::kFloat)) * dt;

    // Pad tensors for chunking.
    torch::Tensor hidden_states_ssm_padded = pad_tensor_4d(hidden_states_ssm);
    torch::Tensor A_padded = pad_tensor_3d(A);
    torch::Tensor B_padded = pad_tensor_4d(B);
    torch::Tensor C_padded = pad_tensor_4d(C);

    const int64_t padded_seq_len = hidden_states_ssm_padded.size(1);
    const int64_t num_chunks = padded_seq_len / kChunkSize;

    // Reshape into chunks.
    torch::Tensor hidden_states_ssm_chunked = hidden_states_ssm_padded.reshape(
        {batch_size, num_chunks, kChunkSize, kNumHeads, kHeadDim});
    torch::Tensor A_chunked =
        A_padded.reshape({batch_size, num_chunks, kChunkSize, kNumHeads});
    torch::Tensor B_chunked = B_padded.reshape(
        {batch_size, num_chunks, kChunkSize, kNumHeads, kSsmStateSize});
    torch::Tensor C_chunked = C_padded.reshape(
        {batch_size, num_chunks, kChunkSize, kNumHeads, kSsmStateSize});

    // Permute A for computation: (batch, num_heads, num_chunks, chunk_size).
    torch::Tensor A_perm = A_chunked.permute({0, 3, 1, 2});
    torch::Tensor A_cumsum = at::cumsum(A_perm, -1);

    // 3a. Intra-chunk computation (diagonal blocks).
    const auto bool_opts =
        at::TensorOptions().dtype(at::kBool).device(hidden_states.device());
    const auto neg_inf = -std::numeric_limits<double>::infinity();

    // segment_sum over a (..., cs) tensor -> (..., cs, cs) lower-triangular
    // exclusive cumulative-sum with -inf above the diagonal.
    auto segment_sum = [&](const torch::Tensor& x) -> torch::Tensor {
        const int64_t cs = x.size(-1);
        auto x_sizes = x.sizes().vec();
        x_sizes.push_back(cs);
        torch::Tensor x_expanded = x.unsqueeze(-1).expand(x_sizes);
        torch::Tensor mask =
            at::tril(at::ones({cs, cs}, bool_opts), /*diagonal=*/-1);
        torch::Tensor x_masked = x_expanded.masked_fill(mask.logical_not(), 0);
        torch::Tensor tensor_segsum = at::cumsum(x_masked, /*dim=*/-2);
        torch::Tensor mask_diag =
            at::tril(at::ones({cs, cs}, bool_opts), /*diagonal=*/0);
        return tensor_segsum.masked_fill(mask_diag.logical_not(), neg_inf);
    };

    torch::Tensor L = at::exp(segment_sum(A_perm));  // (batch, heads, chunks, cs, cs)

    // Attention-like weights G = C^T B over the state dim.
    // G: (batch, chunks, cs_l, cs_s, heads)
    torch::Tensor G = at::einsum("bclhn,bcshn->bclsh", {C_chunked, B_chunked});

    // L -> (batch, chunks, cs_l, cs_s, heads), then M = G * L element-wise.
    torch::Tensor L_perm = L.permute({0, 2, 3, 4, 1});
    torch::Tensor M = G * L_perm;

    // Y_diag: (batch, chunks, cs_l, heads, head_dim)
    torch::Tensor Y_diag =
        at::einsum("bclsh,bcshd->bclhd", {M, hidden_states_ssm_chunked});

    // 3b. States at chunk boundaries.
    // decay_states = exp(A_cumsum[..., -1:] - A_cumsum): (batch, heads, chunks, cs)
    torch::Tensor A_cumsum_last = A_cumsum.narrow(-1, kChunkSize - 1, 1);
    torch::Tensor decay_states = at::exp(A_cumsum_last - A_cumsum);
    torch::Tensor decay_states_perm = decay_states.permute({0, 2, 3, 1});
    // B_decay = B * decay_states[..., None]: (batch, chunks, cs, heads, state)
    torch::Tensor B_decay = B_chunked * decay_states_perm.unsqueeze(-1);
    // states: (batch, chunks, heads, head_dim, state)
    torch::Tensor states =
        at::einsum("bcshd,bcshn->bchdn", {hidden_states_ssm_chunked, B_decay});

    // 3c. Inter-chunk recurrence over the chunk boundaries. This is a fixed
    //     scan over num_chunks; the reference expresses it as a single
    //     contraction of a lower-triangular inter-chunk decay matrix with the
    //     prefixed states, which is retained as an einsum here.
    torch::Tensor previous_states = at::zeros_like(states.narrow(1, 0, 1));
    torch::Tensor states_with_prev = at::cat({previous_states, states}, 1);
    // (batch, chunks+1, heads, head_dim, state)

    // A_chunk_ends = A_cumsum[..., -1]: (batch, heads, chunks), left-padded by 1.
    torch::Tensor A_chunk_ends = A_cumsum.narrow(-1, kChunkSize - 1, 1).squeeze(-1);
    const std::vector<int64_t> chunk_pad{1, 0};
    torch::Tensor A_chunk_ends_padded =
        at::constant_pad_nd(A_chunk_ends, chunk_pad, 0);  // (batch, heads, chunks+1)

    // decay_chunk: (batch, heads, chunks+1, chunks+1) from the same segment_sum.
    torch::Tensor decay_chunk = at::exp(segment_sum(A_chunk_ends_padded));

    // states_with_prev -> (batch, heads, chunks+1, head_dim, state).
    torch::Tensor states_with_prev_perm =
        states_with_prev.permute({0, 2, 1, 3, 4});
    // new_states: (batch, heads, chunks+1, head_dim, state).
    torch::Tensor new_states =
        at::einsum("bhcd,bhdin->bhcin", {decay_chunk, states_with_prev_perm});
    // Drop the last chunk, permute back to (batch, chunks, heads, head_dim, state).
    new_states = new_states.narrow(2, 0, num_chunks);
    torch::Tensor states_final = new_states.permute({0, 2, 1, 3, 4});

    // 3d. State-to-output (off-diagonal blocks).
    torch::Tensor state_decay_out = at::exp(A_cumsum);  // (batch, heads, chunks, cs)
    torch::Tensor state_decay_out_perm = state_decay_out.permute({0, 2, 3, 1});
    // Y_off: (batch, chunks, cs, heads, head_dim)
    torch::Tensor Y_off = at::einsum(
        "bcshn,bchdn,bcsh->bcshd",
        {C_chunked, states_final, state_decay_out_perm});

    // Combine intra- and inter-chunk outputs, reshape, add skip connection.
    torch::Tensor y = Y_diag + Y_off;  // (batch, chunks, cs, heads, head_dim)
    y = y.reshape({batch_size, padded_seq_len, kNumHeads, kHeadDim});
    y = y + D_residual;

    // Remove padding.
    if (pad_size > 0) {
        y = y.narrow(1, 0, seq_len);
    }

    // Reshape and convert back to bf16.
    y = y.reshape({batch_size, seq_len, kIntermediateSize}).to(at::kBFloat16);

    // 4. Gated normalization. Group-wise RMS norm with fp32 variance.
    const int64_t group_size = kIntermediateSize / kNGroups;
    torch::Tensor y_grouped =
        y.view({batch_size, seq_len, kNGroups, group_size});
    torch::Tensor variance =
        y_grouped.to(at::kFloat).pow(2).mean(-1, /*keepdim=*/true);
    torch::Tensor y_normed =
        y_grouped * at::rsqrt(variance + layer_norm_epsilon);
    y_normed =
        y_normed.view({batch_size, seq_len, kIntermediateSize}).to(at::kBFloat16);
    y_normed = y_normed * norm_weight;
    // silu gate: y_normed * (gate * sigmoid(gate)).
    y = y_normed * (gate * at::sigmoid(gate));

    // 5. Output projection: y @ out_proj_weight^T -> (B, L, hidden_size).
    torch::Tensor result = at::matmul(y, out_proj_weight.t());

    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Mamba2 selective scan via ATen (input projection / causal depthwise "
          "conv1d + SiLU / softplus + clamp discretization / chunked selective "
          "scan with segment-sum decay masks, intra-chunk diagonal, inter-chunk "
          "recurrence, and off-diagonal state-to-output / gated RMS norm / output "
          "projection; no custom CUDA kernels). bf16 I/O with fp32 scan compute; "
          "destination-passing style.");
}
