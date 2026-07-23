#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kernel.h"
#include <vector>
#include <string>
#include <cmath>

// Audio relative-position attention reproduced op-for-op from the reference
// run(). Audio is processed in fixed CHUNK_SIZE query blocks, each attending to
// an overlapping CONTEXT_SIZE window of keys/values built by left/right padding
// and an unfold. Attention carries a content term (query @ key) plus a
// relative-position term (query @ projected sinusoidal embedding) with a
// shift that aligns each query position to its relative offset; the logits are
// soft-capped through a tanh, masked by the combined input-validity and local
// causal mask, softmaxed, and used to weight the value context.
//
// Fixed constants from the spec:
//   NUM_HEADS    = 8
//   HEAD_DIM     = 64
//   CHUNK_SIZE   = 32
//   MAX_BACKWARD = 127
//   MAX_FORWARD  = 0
//   CONTEXT_SIZE = 160 (== MAX_BACKWARD + MAX_FORWARD + CHUNK_SIZE + 1)
//   hidden_size  = NUM_HEADS * HEAD_DIM = 512
//
// Per definition.json (insertion order):
//   hidden_states              : [B, seq_len, hidden_size]        fp32
//   mask                       : [B, seq_len]                     bool (True = padded)
//   q_proj_weight              : [hidden_size, hidden_size]       fp32
//   k_proj_weight              : [hidden_size, hidden_size]       fp32
//   v_proj_weight              : [hidden_size, hidden_size]       fp32
//   pos_proj_weight            : [hidden_size, hidden_size]       fp32
//   per_dim_scale              : [HEAD_DIM]                       fp32
//   inv_timescales             : [num_timescales]                 fp32
//   attention_logits_soft_cap  : scalar                           fp32 -> double
//   context_vectors            : [B, seq_len, NUM_HEADS, HEAD_DIM] fp32 DPS
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels.
// at::matmul / at::bmm / at::softmax / at::tanh / at::softplus / at::where and
// the unfold / movedim / pad / reshape / narrow shape plumbing match the
// reference's ops with the native fp32 precision profile. The output is
// written in place into the pre-allocated destination (destination-passing
// style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

// Pad along the time dimension (dim 1) with pad_left / pad_right zero rows,
// matching the reference's pad_dim1 (new_zeros concatenation, NOT a reflect /
// replicate pad). at::constant_pad_nd pads from the last dimension inward, so
// the leading 2*ndim-2 zeros leave the trailing tail dims untouched and place
// the time pad on dim 1.
static torch::Tensor pad_dim1(const torch::Tensor& x, int64_t pad_left, int64_t pad_right) {
    std::vector<int64_t> pad(2 * x.dim(), 0);
    // Dimension 1 sits (ndim - 1 - 1) positions from the last dim; its pad pair
    // occupies indices [2*(ndim-2), 2*(ndim-2)+1] in the last-first pad vector.
    const int64_t slot = 2 * (x.dim() - 2);
    pad[slot] = pad_left;
    pad[slot + 1] = pad_right;
    return at::constant_pad_nd(x, pad, 0.0);
}

void run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& mask,
    const torch::Tensor& q_proj_weight,
    const torch::Tensor& k_proj_weight,
    const torch::Tensor& v_proj_weight,
    const torch::Tensor& pos_proj_weight,
    const torch::Tensor& per_dim_scale,
    const torch::Tensor& inv_timescales,
    double attention_logits_soft_cap,
    torch::Tensor context_vectors
) {
    check_fp32_cuda(hidden_states, "hidden_states");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be bool, got ", mask.dtype());
    check_fp32_cuda(q_proj_weight, "q_proj_weight");
    check_fp32_cuda(k_proj_weight, "k_proj_weight");
    check_fp32_cuda(v_proj_weight, "v_proj_weight");
    check_fp32_cuda(pos_proj_weight, "pos_proj_weight");
    check_fp32_cuda(per_dim_scale, "per_dim_scale");
    check_fp32_cuda(inv_timescales, "inv_timescales");
    check_fp32_cuda(context_vectors, "context_vectors");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, seq_len, hidden_size], got ",
                hidden_states.dim(), "D");
    TORCH_CHECK(mask.dim() == 2,
                "mask must be 2D [B, seq_len], got ", mask.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(hidden_states.device());

    constexpr int64_t kNumHeads = 8;
    constexpr int64_t kHeadDim = 64;
    constexpr int64_t kChunkSize = 32;
    constexpr int64_t kMaxBackward = 127;
    constexpr int64_t kMaxForward = 0;
    constexpr int64_t kContextSize = 160;

    const int64_t batch_size = hidden_states.size(0);
    const int64_t q_time = hidden_states.size(1);

    // Query scaling: head_dim ** -0.5 times the reciprocal of softplus(0).
    const double q_scale_base = std::pow(static_cast<double>(kHeadDim), -0.5);
    auto softplus_0 = at::softplus(at::zeros({}, hidden_states.options()));
    auto r_softplus_0 = at::reciprocal(softplus_0);
    auto q_scale = static_cast<double>(q_scale_base) * r_softplus_0;  // 0-dim tensor

    // Build the local causal validity mask over [CHUNK_SIZE, CONTEXT_SIZE].
    // lower bound: tril(ones(CONTEXT_SIZE, CHUNK_SIZE), 0).T;  upper bound:
    // tril(ones(CHUNK_SIZE, CONTEXT_SIZE), MAX_BACKWARD + MAX_FORWARD); the two
    // bands are AND-ed (via boolean multiply) into a single valid window.
    auto bool_opts = hidden_states.options().dtype(torch::kBool);
    auto lower_causal_mask =
        at::tril(at::ones({kContextSize, kChunkSize}, bool_opts), 0).t();
    auto upper_causal_mask =
        at::tril(at::ones({kChunkSize, kContextSize}, bool_opts), kMaxBackward + kMaxForward);
    auto local_causal_valid_mask =
        at::ones({kChunkSize, kContextSize}, bool_opts) * lower_causal_mask * upper_causal_mask;

    // Project to Q / K / V via x @ W^T.
    auto query_states = at::matmul(hidden_states, q_proj_weight.t());
    auto key_states = at::matmul(hidden_states, k_proj_weight.t());
    auto value_states = at::matmul(hidden_states, v_proj_weight.t());

    // Reshape to [B, T, NUM_HEADS, HEAD_DIM].
    auto qkv_shape = std::vector<int64_t>{batch_size, q_time, kNumHeads, kHeadDim};
    query_states = query_states.reshape(qkv_shape).contiguous();
    key_states = key_states.reshape(qkv_shape).contiguous();
    value_states = value_states.reshape(qkv_shape).contiguous();

    // Per-dimension query scaling: q * q_scale * softplus(per_dim_scale) broadcast
    // over [1, 1, 1, HEAD_DIM].
    auto per_dim_scale_sp = at::softplus(per_dim_scale);
    auto per_dim_scale_sp_broadcast =
        per_dim_scale_sp.view(std::vector<int64_t>{1, 1, 1, kHeadDim});
    query_states = query_states * q_scale * per_dim_scale_sp_broadcast;

    // convert_to_block: pad the time axis up to a whole number of CHUNK_SIZE
    // blocks, then reshape into [B, num_blocks, CHUNK_SIZE, ...].
    auto convert_to_block = [&](const torch::Tensor& hs) {
        const int64_t b = hs.size(0);
        const int64_t t = hs.size(1);
        const int64_t num_blocks = (t + kChunkSize - 1) / kChunkSize;
        const int64_t padding_len = num_blocks * kChunkSize - t;
        torch::Tensor padded = hs;
        if (padding_len > 0) {
            padded = pad_dim1(padded, 0, padding_len);
        }
        std::vector<int64_t> permute_dims{b, num_blocks, kChunkSize};
        for (int64_t d = 2; d < padded.dim(); ++d) {
            permute_dims.push_back(padded.size(d));
        }
        return padded.reshape(permute_dims).contiguous();
    };

    // extract_block_context: left/right pad the time axis, unfold a sliding
    // CONTEXT_SIZE window with CHUNK_SIZE stride, then (for tensors carrying
    // head/dim tails) move the unfolded window axis to position 2.
    auto extract_block_context = [&](const torch::Tensor& hs_in) {
        const int64_t pad_left = kMaxBackward;
        const int64_t pad_right = kMaxForward + kChunkSize - 1;
        auto hs = pad_dim1(hs_in, pad_left, pad_right);
        auto x_unfolded = hs.unfold(/*dimension=*/1, /*size=*/kContextSize, /*step=*/kChunkSize);
        if (hs.dim() > 2 && x_unfolded.dim() > 3) {
            x_unfolded = at::movedim(x_unfolded, /*source=*/-1, /*destination=*/2);
        }
        return x_unfolded.contiguous();
    };

    // get_timing_signal_1d_pos: sinusoidal encoding of integer positions; sin
    // and cos of position * inv_timescales concatenated on the last axis.
    auto get_timing_signal_1d_pos = [&](const torch::Tensor& position) {
        auto pos = position.to(torch::kFloat32).unsqueeze(-1);
        auto scaled_time = pos * inv_timescales.unsqueeze(0).unsqueeze(0);
        auto timing_signal =
            at::cat({at::sin(scaled_time), at::cos(scaled_time)}, /*dim=*/-1);
        return timing_signal;
    };

    // Convert query into blocks and gather key / value context windows.
    auto query_blocks = convert_to_block(query_states);          // [B, U, W, NH, HD]
    auto key_blocks = extract_block_context(key_states);         // [B, U, C, NH, HD]
    auto value_blocks = extract_block_context(value_states);     // [B, U, C, NH, HD]
    const int64_t num_query_blocks = query_blocks.size(1);

    // Validity mask blocks: ~mask, windowed and folded to [B, U, CONTEXT_SIZE].
    auto original_valid_mask = at::logical_not(mask);
    auto extracted_valid_mask_blocks = extract_block_context(original_valid_mask);
    if (extracted_valid_mask_blocks.dim() == 4) {
        extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(
            std::vector<int64_t>{batch_size, num_query_blocks, kContextSize});
    }

    // Combined where-condition: input validity AND local causality, broadcast
    // over heads and query positions.
    auto condition_from_input_validity =
        extracted_valid_mask_blocks.unsqueeze(1).unsqueeze(-2);  // [B, 1, U, 1, C]
    auto condition_from_causality =
        local_causal_valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0);  // [1, 1, 1, W, C]
    auto final_condition_for_where =
        at::logical_and(condition_from_input_validity, condition_from_causality);

    // Relative position offsets MAX_BACKWARD .. -MAX_FORWARD (descending) and
    // their projected sinusoidal embedding.
    auto pos_indices =
        at::arange(kMaxBackward, -kMaxForward - 1, -1, hidden_states.options())
            .unsqueeze(0);  // [1, max_span_plus_1]
    const int64_t max_span_plus_1 = pos_indices.size(1);

    auto sin_emb_timing_signal = get_timing_signal_1d_pos(pos_indices);
    auto projected_sin_emb = at::matmul(sin_emb_timing_signal, pos_proj_weight.t());
    auto sin_emb =
        projected_sin_emb.reshape(std::vector<int64_t>{1, max_span_plus_1, kNumHeads, kHeadDim})
            .squeeze(0);  // [max_span_plus_1, NH, HD]

    // Content term: query @ key^T per (head, block) window. [B, NH, U, W, C].
    auto queries_p = query_blocks.permute({0, 3, 1, 2, 4});  // [B, NH, U, W, HD]
    auto keys_p_t = key_blocks.permute({0, 3, 1, 4, 2});     // [B, NH, U, HD, C]
    auto term_ac = at::matmul(queries_p, keys_p_t);          // [B, NH, U, W, C]

    // Position term: query @ position-embedding^T, then a relative shift.
    auto q_permuted = query_blocks.permute({0, 3, 1, 2, 4});  // [B, NH, U, W, HD]
    auto s_permuted = sin_emb.permute({1, 2, 0});             // [NH, HD, max_span_plus_1]
    auto q_reshaped = q_permuted.reshape(
        std::vector<int64_t>{batch_size, kNumHeads, num_query_blocks * kChunkSize, kHeadDim});
    auto term_bd_unshifed_matmul = at::matmul(q_reshaped, s_permuted);
    auto term_bd_unshifed = term_bd_unshifed_matmul.reshape(std::vector<int64_t>{
        batch_size, kNumHeads, num_query_blocks, kChunkSize, max_span_plus_1});

    // relative_shift: right-pad the last axis to (CONTEXT_SIZE + 1), flatten the
    // (W, C+1) pair, drop the trailing W positions, then refold to [.., W, C].
    // The pad / reshape / narrow sequence skews each query row by its offset so
    // relative positions line up against the absolute key window.
    const int64_t pad_amount_last_dim = (kContextSize + 1) - max_span_plus_1;
    auto term_bd_padded = at::constant_pad_nd(
        term_bd_unshifed, std::vector<int64_t>{0, pad_amount_last_dim}, 0.0);
    auto term_bd_reshaped = term_bd_padded.reshape(std::vector<int64_t>{
        batch_size, kNumHeads, num_query_blocks, kChunkSize * (kContextSize + 1)});
    auto term_bd_sliced =
        term_bd_reshaped.narrow(/*dim=*/3, /*start=*/0, /*length=*/kChunkSize * kContextSize);
    auto term_bd_shifted = term_bd_sliced.reshape(std::vector<int64_t>{
        batch_size, kNumHeads, num_query_blocks, kChunkSize, kContextSize});

    auto logits = term_ac + term_bd_shifted;

    // Soft-cap the logits: scale into the tanh's linear region, squash, rescale.
    auto softcap_val = at::full({}, attention_logits_soft_cap, hidden_states.options());
    logits = logits / softcap_val;
    logits = at::tanh(logits);
    logits = logits * softcap_val;

    // Mask out invalid positions with the fp32 lowest finite value.
    auto neg_min = at::full(
        {}, std::numeric_limits<float>::lowest(), hidden_states.options());
    logits = at::where(final_condition_for_where, logits, neg_min);

    // Softmax over the key-context axis (native fp32), then weight the values.
    auto probabilities = at::softmax(logits, /*dim=*/-1);  // [B, NH, U, W, C]

    const int64_t b_dim = probabilities.size(0);
    const int64_t n_dim = probabilities.size(1);
    const int64_t u_dim = probabilities.size(2);
    const int64_t w_dim = probabilities.size(3);
    const int64_t c_dim = probabilities.size(4);
    const int64_t h_dim = value_blocks.size(-1);

    // prob[B,U,NH,W,C] @ value[B,U,NH,C,HD] batched over (B*U*NH).
    auto prob_bun = probabilities.permute({0, 2, 1, 3, 4})
                        .reshape(std::vector<int64_t>{-1, w_dim, c_dim});
    auto v_bun = value_blocks.permute({0, 1, 3, 2, 4})
                     .reshape(std::vector<int64_t>{-1, c_dim, h_dim});
    auto result_bmm = at::bmm(prob_bun, v_bun);  // [B*U*NH, W, HD]

    auto out = result_bmm
                   .reshape(std::vector<int64_t>{b_dim, u_dim, n_dim, w_dim, h_dim})
                   .permute({0, 1, 3, 2, 4});  // [B, U, W, NH, HD]
    out = out.reshape(std::vector<int64_t>{
        batch_size, num_query_blocks * kChunkSize, kNumHeads, kHeadDim});
    out = out.narrow(/*dim=*/1, /*start=*/0, /*length=*/q_time);  // drop block padding

    TORCH_CHECK(out.sizes() == context_vectors.sizes(),
                "output shape ", out.sizes(),
                " does not match pre-allocated ", context_vectors.sizes());
    context_vectors.copy_(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Audio relative-position attention via pure ATen dispatch: chunked "
          "local windows (convert_to_block / extract_block_context via unfold + "
          "movedim), per-dimension softplus query scaling, a content term "
          "(query @ key) plus a relative-shifted position term (query @ "
          "projected sinusoidal embedding), tanh soft-capping, a combined "
          "input-validity / local-causal mask via at::where, softmax over the "
          "key window, and a batched value-weighted sum; no custom CUDA "
          "kernels; fp32; destination-passing style (context_vectors).");
}
