#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <limits>
#include <string>

// Chunk-based Gated Delta Rule linear attention.
//
// Reference (problem_060_chunk_gated_delta_rule_linear_attention), reproduced
// op-for-op. Sequences are processed in chunks of 64 tokens with an L2-normed,
// scaled query / key, a delta-rule intra-chunk attention solved by a fixed
// sequential row-update loop, and an inter-chunk recurrent state carried across
// chunks by a second sequential loop.
//
// Pinned constants (definition.json axes):
//   num_k_heads = 4 ; num_v_heads = 16 ; head_k_dim = 128 ; head_v_dim = 128
//   chunk_size = 64
//
// Per definition.json (input insertion order, then the single output):
//   query  : [batch, num_k_heads, seq, head_k_dim]   bf16
//   key    : [batch, num_k_heads, seq, head_k_dim]   bf16
//   value  : [batch, num_v_heads, seq, head_v_dim]   bf16
//   g      : [batch, num_v_heads, seq]               bf16
//   beta   : [batch, num_v_heads, seq]               bf16
//   scale  : scalar fp32
//   output : [batch, seq, num_v_heads, head_v_dim]   bf16 (DPS)
//
// Dtype profile (Rule 8): bf16 I/O. The L2 normalization of query / key runs in
// the input bf16 dtype (the reference calls l2norm BEFORE the `.to(float32)`
// casts, so the (x*x).sum reduction is bf16), and everything downstream of the
// explicit `query/key/value/beta/g = ....to(torch.float32)` casts is fp32: the
// scaling, padding, beta-weighting, cumulative-decay masks, the delta-rule
// sequential row update, all matmuls, and the recurrent-state scan. The final
// output is cast back to the input dtype with `.to(initial_dtype)`.
//
// Two sequential loops are ported as C++ host for-loops, exactly as in the
// reference:
//   * Inner: `for i in range(1, chunk_size)` solves the lower-triangular delta
//     system one row at a time, reading rows < i (cloned) and writing row i.
//     The slices are taken with .narrow / .select on the last two dims of the
//     [batch, heads, chunks, cs, cs] attn tensor.
//   * Outer: `for chunk_idx in range(num_chunks)` carries recurrent_state across
//     chunk boundaries, writing one chunk of output per iteration.
// num_chunks and chunk_size are static shapes, so both loops have fixed,
// data-independent trip counts (legal host loops on the cuda_cpp surface;
// no custom kernel, current stream).
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. Output
// is written in place into the pre-allocated destination.

static void check_bf16_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bfloat16, got ", t.dtype());
}

// L2 normalization over the last dim, computed in the tensor's own dtype.
static torch::Tensor l2norm(const torch::Tensor& x, double eps) {
    torch::Tensor inv_norm =
        at::rsqrt((x * x).sum(-1, /*keepdim=*/true) + eps);
    return x * inv_norm;
}

void run(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    double scale,
    torch::Tensor output
) {
    check_bf16_cuda(query, "query");
    check_bf16_cuda(key, "key");
    check_bf16_cuda(value, "value");
    check_bf16_cuda(g, "g");
    check_bf16_cuda(beta, "beta");
    check_bf16_cuda(output, "output");

    TORCH_CHECK(query.dim() == 4,
                "query must be 4D [batch, num_k_heads, seq, head_k_dim], got ",
                query.dim(), "D");
    TORCH_CHECK(value.dim() == 4,
                "value must be 4D [batch, num_v_heads, seq, head_v_dim], got ",
                value.dim(), "D");
    TORCH_CHECK(g.dim() == 3, "g must be 3D [batch, num_v_heads, seq], got ",
                g.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(query.device());

    constexpr int64_t kChunkSize = 64;

    const int64_t batch_size = key.size(0);
    const int64_t num_k_heads = key.size(1);
    const int64_t sequence_length = key.size(2);
    const int64_t k_head_dim = key.size(3);
    const int64_t num_v_heads = value.size(1);
    const int64_t v_head_dim = value.size(-1);
    const int64_t heads_ratio = num_v_heads / num_k_heads;

    // L2 normalization (bf16), then cast to fp32 for numerical stability.
    torch::Tensor q = l2norm(query, 1e-6);
    torch::Tensor k = l2norm(key, 1e-6);

    q = q.to(at::kFloat);
    k = k.to(at::kFloat);
    torch::Tensor v = value.to(at::kFloat);
    torch::Tensor beta_f = beta.to(at::kFloat);
    torch::Tensor g_f = g.to(at::kFloat);

    // Apply scaling to queries.
    q = q * scale;

    // Pad sequence to a multiple of chunk_size on the seq axis.
    const int64_t pad_size =
        (kChunkSize - sequence_length % kChunkSize) % kChunkSize;
    if (pad_size > 0) {
        // q/k/v: pad the seq axis (second-from-last of a 4D tensor). F.pad
        // (0,0, 0,pad) pads the last two dims; constant_pad_nd takes the same
        // last-dim-first low/high pairs.
        const std::vector<int64_t> pad_qkv{0, 0, 0, pad_size};
        q = at::constant_pad_nd(q, pad_qkv, 0);
        k = at::constant_pad_nd(k, pad_qkv, 0);
        v = at::constant_pad_nd(v, pad_qkv, 0);
        // beta/g: pad the last (seq) dim.
        const std::vector<int64_t> pad_bg{0, pad_size};
        beta_f = at::constant_pad_nd(beta_f, pad_bg, 0);
        g_f = at::constant_pad_nd(g_f, pad_bg, 0);
    }

    const int64_t total_sequence_length = sequence_length + pad_size;
    const int64_t num_chunks = total_sequence_length / kChunkSize;

    // Beta-weighted values and keys. v_beta keeps the v-head layout.
    torch::Tensor v_beta = v * beta_f.unsqueeze(-1);
    // k_beta expands the k-heads up to v-heads (group repeat), then beta-weights.
    torch::Tensor k_beta = k.unsqueeze(1)
                               .expand({batch_size, heads_ratio, num_k_heads,
                                        total_sequence_length, k_head_dim})
                               .reshape({batch_size, num_v_heads,
                                         total_sequence_length, k_head_dim});
    k_beta = k_beta * beta_f.unsqueeze(-1);

    // Reshape to chunks. query/key expand to v-heads then split the seq axis.
    torch::Tensor query_exp = q.unsqueeze(1)
                                  .expand({batch_size, heads_ratio, num_k_heads,
                                           total_sequence_length, k_head_dim})
                                  .reshape({batch_size, num_v_heads,
                                            total_sequence_length, k_head_dim});
    torch::Tensor query_chunks = query_exp.reshape(
        {batch_size, num_v_heads, num_chunks, kChunkSize, k_head_dim});

    torch::Tensor key_exp = k.unsqueeze(1)
                                .expand({batch_size, heads_ratio, num_k_heads,
                                         total_sequence_length, k_head_dim})
                                .reshape({batch_size, num_v_heads,
                                          total_sequence_length, k_head_dim});
    torch::Tensor key_chunks = key_exp.reshape(
        {batch_size, num_v_heads, num_chunks, kChunkSize, k_head_dim});

    torch::Tensor value_chunks = v.reshape(
        {batch_size, num_v_heads, num_chunks, kChunkSize, v_head_dim});
    torch::Tensor k_beta_chunks = k_beta.reshape(
        {batch_size, num_v_heads, num_chunks, kChunkSize, k_head_dim});
    torch::Tensor v_beta_chunks = v_beta.reshape(
        {batch_size, num_v_heads, num_chunks, kChunkSize, v_head_dim});
    torch::Tensor g_chunks =
        g_f.reshape({batch_size, num_v_heads, num_chunks, kChunkSize});

    // Cumulative decay within chunks.
    torch::Tensor g_cumsum = g_chunks.cumsum(-1);

    // Decay mask = exp(g_cumsum[..., :, None] - g_cumsum[..., None, :]), tril.
    torch::Tensor decay_mask =
        (g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)).exp();
    decay_mask = at::tril(decay_mask);

    const auto bool_opts =
        at::TensorOptions().dtype(at::kBool).device(query.device());

    // Causal mask: upper triangular including the diagonal.
    torch::Tensor causal_mask = at::triu(
        at::ones({kChunkSize, kChunkSize}, bool_opts), /*diagonal=*/0);

    // Intra-chunk attention seed: attn = -(k_beta @ key^T) * decay_mask, with
    // the causal (>= diagonal) entries zeroed.
    torch::Tensor attn =
        -at::matmul(k_beta_chunks, key_chunks.transpose(-1, -2)) * decay_mask;
    attn = attn.masked_fill(causal_mask, 0);

    // Sequential lower-triangular solve. For each row i (1..cs-1):
    //   row = attn[..., i, :i]              (length i)
    //   sub = attn[..., :i, :i]             (i x i)
    //   attn[..., i, :i] = row + (row[..., None] * sub).sum(-2)
    for (int64_t i = 1; i < kChunkSize; ++i) {
        torch::Tensor row =
            attn.select(-2, i).narrow(-1, 0, i).clone();  // [..., i]
        torch::Tensor sub =
            attn.narrow(-2, 0, i).narrow(-1, 0, i).clone();  // [..., i, i]
        torch::Tensor updated =
            row + (row.unsqueeze(-1) * sub).sum(-2);  // [..., i]
        attn.select(-2, i).narrow(-1, 0, i).copy_(updated);
    }

    // Add identity.
    torch::Tensor eye =
        at::eye(kChunkSize, at::TensorOptions()
                                .dtype(attn.dtype())
                                .device(attn.device()));
    attn = attn + eye;

    // Transform values and cumulative key decay.
    torch::Tensor value_transformed = at::matmul(attn, v_beta_chunks);
    torch::Tensor k_cumdecay =
        at::matmul(attn, k_beta_chunks * g_chunks.exp().unsqueeze(-1));

    // Recurrent state and output accumulator (fp32).
    const auto f32_opts =
        at::TensorOptions().dtype(at::kFloat).device(value.device());
    torch::Tensor recurrent_state =
        at::zeros({batch_size, num_v_heads, k_head_dim, v_head_dim}, f32_opts);
    torch::Tensor out_chunks = at::zeros(
        {batch_size, num_v_heads, num_chunks, kChunkSize, v_head_dim}, f32_opts);

    // Strictly-upper-triangular mask (excludes the diagonal).
    torch::Tensor upper_mask = at::triu(
        at::ones({kChunkSize, kChunkSize}, bool_opts), /*diagonal=*/1);

    // Sequential inter-chunk recurrence carrying recurrent_state across chunks.
    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        torch::Tensor q_chunk = query_chunks.select(2, chunk_idx);
        torch::Tensor k_chunk = key_chunks.select(2, chunk_idx);
        torch::Tensor v_transformed_chunk = value_transformed.select(2, chunk_idx);
        torch::Tensor k_cumdecay_chunk = k_cumdecay.select(2, chunk_idx);
        torch::Tensor g_chunk = g_chunks.select(2, chunk_idx);  // [batch, heads, cs]
        torch::Tensor decay_mask_chunk = decay_mask.select(2, chunk_idx);

        // Intra-chunk attention with strictly-upper-triangular entries zeroed.
        torch::Tensor attn_intra =
            at::matmul(q_chunk, k_chunk.transpose(-1, -2)) * decay_mask_chunk;
        attn_intra = attn_intra.masked_fill(upper_mask, 0);

        // Value correction from the recurrent state.
        torch::Tensor v_prime = at::matmul(k_cumdecay_chunk, recurrent_state);
        torch::Tensor v_new = v_transformed_chunk - v_prime;

        // Inter-chunk attention.
        torch::Tensor attn_inter =
            at::matmul(q_chunk * g_chunk.exp().unsqueeze(-1), recurrent_state);

        // Combine and write this chunk's output.
        out_chunks.select(2, chunk_idx)
            .copy_(attn_inter + at::matmul(attn_intra, v_new));

        // Update the recurrent state: decay by the chunk-final gate, then add
        // the token-decayed key^T @ v_new outer product.
        torch::Tensor g_last = g_chunk.select(-1, kChunkSize - 1);  // [batch, heads]
        torch::Tensor chunk_decay = g_last.exp().unsqueeze(-1).unsqueeze(-1);
        recurrent_state = recurrent_state * chunk_decay;

        torch::Tensor token_decay =
            (g_last.unsqueeze(-1) - g_chunk).exp().unsqueeze(-1);  // [batch, heads, cs, 1]
        recurrent_state = recurrent_state +
            at::matmul((k_chunk * token_decay).transpose(-1, -2), v_new);
    }

    // Reshape output and remove padding.
    torch::Tensor result = out_chunks.reshape(
        {batch_size, num_v_heads, total_sequence_length, v_head_dim});
    result = result.narrow(2, 0, sequence_length);

    // Transpose to [batch, seq, num_v_heads, head_v_dim] and cast back to bf16.
    result = result.transpose(1, 2).contiguous().to(at::kBFloat16);

    TORCH_CHECK(result.sizes() == output.sizes(),
                "output shape ", result.sizes(),
                " does not match pre-allocated ", output.sizes());
    output.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Chunk-based gated delta-rule linear attention via ATen (L2-normed "
          "scaled q/k, fixed sequential lower-triangular delta solve, chunked "
          "intra/inter attention with a recurrent-state scan over chunks; no "
          "custom CUDA kernels). bf16 I/O with fp32 scan compute; "
          "destination-passing style.");
}
