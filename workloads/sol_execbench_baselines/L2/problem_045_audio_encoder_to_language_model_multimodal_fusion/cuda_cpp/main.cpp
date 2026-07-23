#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>
#include <cmath>

// End-to-end audio-to-language multimodal fusion: a CTC-style encoder, a
// windowed Q-Former cross-attention downsampler with learnable queries, and a
// per-batch scatter of the resulting audio embeddings into the language-model
// token embeddings. Reproduces the reference run() op-for-op via pure ATen /
// cuBLAS dispatch -- no custom device kernels.
//
// Pinned axes from the problem definition:
//   encoder_input_dim     = 80
//   encoder_hidden_dim    = 512
//   encoder_output_dim    = 1024
//   projector_hidden_size = 1024
//   window_size           = 15
//   num_queries           = 40
//   text_hidden_size      = 4096
//   qformer_num_heads     = 16
//   qformer_head_dim      = projector_hidden_size / qformer_num_heads = 64
//
// Per definition.json (insertion order):
//   inputs (all bfloat16 except audio_token_positions which is int64):
//     input_features          : [batch, audio_seq_len, encoder_input_dim]
//     lm_embeddings           : [batch, text_seq_len, text_hidden_size]
//     audio_token_positions   : [batch, num_audio_tokens]                int64
//     encoder_input_weight    : [encoder_hidden_dim, encoder_input_dim]
//     encoder_input_bias      : [encoder_hidden_dim]
//     encoder_out_weight      : [encoder_output_dim, encoder_hidden_dim]
//     encoder_out_bias        : [encoder_output_dim]
//     encoder_out_mid_weight  : [encoder_hidden_dim, encoder_output_dim]
//     encoder_out_mid_bias    : [encoder_hidden_dim]
//     learnable_queries       : [1, num_queries, projector_hidden_size]
//     qformer_q_proj_weight   : [projector_hidden_size, projector_hidden_size]
//     qformer_q_proj_bias     : [projector_hidden_size]
//     qformer_k_proj_weight   : [projector_hidden_size, encoder_hidden_dim]
//     qformer_k_proj_bias     : [projector_hidden_size]
//     qformer_v_proj_weight   : [projector_hidden_size, encoder_hidden_dim]
//     qformer_v_proj_bias     : [projector_hidden_size]
//     qformer_out_proj_weight : [projector_hidden_size, projector_hidden_size]
//     qformer_out_proj_bias   : [projector_hidden_size]
//     projector_weight        : [text_hidden_size, projector_hidden_size]
//     projector_bias          : [text_hidden_size]
//   output (bfloat16, DPS pre-alloc, written in place):
//     fused_embeddings        : [batch, text_seq_len, text_hidden_size]
//
// Math the reference computes:
//   Stage 1 (encoder, entirely fp32 -- every linear upcasts inputs / weights /
//   bias to fp32 before the matmul):
//     hidden     = linear(input_features, encoder_input_weight, encoder_input_bias)
//     hidden_mid = linear(hidden, encoder_out_weight, encoder_out_bias)
//     feedback   = linear(softmax(hidden_mid, -1), encoder_out_mid_weight, encoder_out_mid_bias)
//     encoder_output = hidden + feedback                    [batch, audio_seq_len, 512]
//   Stage 2 (windowed Q-Former, fp32):
//     nblocks = ceil(audio_seq_len / window_size); pad = nblocks*window_size - audio_seq_len
//     pad encoder_output's seq dim to a window boundary, reshape into
//       [batch*nblocks, window_size, 512] windows
//     broadcast learnable_queries to [batch*nblocks, num_queries, 1024]
//     Q = linear(queries, q_proj); K = linear(windows, k_proj); V = linear(windows, v_proj)
//     reshape to heads, scaled dot-product attention (scale = 1/sqrt(head_dim)),
//       softmax over keys, weighted sum of V, merge heads
//     query_output = linear(attn_output, out_proj)          [batch*nblocks, num_queries, 1024]
//     reshape to [batch, nblocks*num_queries, 1024]
//     audio_embeddings = linear(query_output, projector)    [batch, nblocks*num_queries, 4096]
//   Stage 3 (per-batch scatter, fp32 accumulator):
//     fused = lm_embeddings.float().clone()
//     for b in range(batch):
//         fused[b].index_copy_(0, audio_token_positions[b],
//                              audio_embeddings[b, :num_audio_tokens, :])
//     return fused.bfloat16()
//
// Precision path: the reference upcasts every operand of every linear to fp32
// and runs the encoder, the Q-Former attention, the projector, and the Stage-3
// fused accumulator entirely in fp32; the inputs arrive as bf16 and the only
// narrowing is the single trailing .to(bfloat16) on the final fused tensor.
// This port matches that exactly -- all compute is fp32, the lone bf16 cast is
// the final copy into the output. The per-batch host loop issues index_copy_
// only on the current PyTorch stream and mirrors the reference's scatter.

static void check_dtype_cuda(const torch::Tensor& t,
                             torch::ScalarType dt,
                             const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == dt, name, " has wrong dtype, got ", t.dtype());
}

void run(
    const torch::Tensor& input_features,
    const torch::Tensor& lm_embeddings,
    const torch::Tensor& audio_token_positions,
    const torch::Tensor& encoder_input_weight,
    const torch::Tensor& encoder_input_bias,
    const torch::Tensor& encoder_out_weight,
    const torch::Tensor& encoder_out_bias,
    const torch::Tensor& encoder_out_mid_weight,
    const torch::Tensor& encoder_out_mid_bias,
    const torch::Tensor& learnable_queries,
    const torch::Tensor& qformer_q_proj_weight,
    const torch::Tensor& qformer_q_proj_bias,
    const torch::Tensor& qformer_k_proj_weight,
    const torch::Tensor& qformer_k_proj_bias,
    const torch::Tensor& qformer_v_proj_weight,
    const torch::Tensor& qformer_v_proj_bias,
    const torch::Tensor& qformer_out_proj_weight,
    const torch::Tensor& qformer_out_proj_bias,
    const torch::Tensor& projector_weight,
    const torch::Tensor& projector_bias,
    torch::Tensor fused_embeddings
) {
    check_dtype_cuda(input_features, torch::kBFloat16, "input_features");
    check_dtype_cuda(lm_embeddings, torch::kBFloat16, "lm_embeddings");
    check_dtype_cuda(audio_token_positions, torch::kInt64, "audio_token_positions");
    check_dtype_cuda(encoder_input_weight, torch::kBFloat16, "encoder_input_weight");
    check_dtype_cuda(encoder_input_bias, torch::kBFloat16, "encoder_input_bias");
    check_dtype_cuda(encoder_out_weight, torch::kBFloat16, "encoder_out_weight");
    check_dtype_cuda(encoder_out_bias, torch::kBFloat16, "encoder_out_bias");
    check_dtype_cuda(encoder_out_mid_weight, torch::kBFloat16, "encoder_out_mid_weight");
    check_dtype_cuda(encoder_out_mid_bias, torch::kBFloat16, "encoder_out_mid_bias");
    check_dtype_cuda(learnable_queries, torch::kBFloat16, "learnable_queries");
    check_dtype_cuda(qformer_q_proj_weight, torch::kBFloat16, "qformer_q_proj_weight");
    check_dtype_cuda(qformer_q_proj_bias, torch::kBFloat16, "qformer_q_proj_bias");
    check_dtype_cuda(qformer_k_proj_weight, torch::kBFloat16, "qformer_k_proj_weight");
    check_dtype_cuda(qformer_k_proj_bias, torch::kBFloat16, "qformer_k_proj_bias");
    check_dtype_cuda(qformer_v_proj_weight, torch::kBFloat16, "qformer_v_proj_weight");
    check_dtype_cuda(qformer_v_proj_bias, torch::kBFloat16, "qformer_v_proj_bias");
    check_dtype_cuda(qformer_out_proj_weight, torch::kBFloat16, "qformer_out_proj_weight");
    check_dtype_cuda(qformer_out_proj_bias, torch::kBFloat16, "qformer_out_proj_bias");
    check_dtype_cuda(projector_weight, torch::kBFloat16, "projector_weight");
    check_dtype_cuda(projector_bias, torch::kBFloat16, "projector_bias");
    check_dtype_cuda(fused_embeddings, torch::kBFloat16, "fused_embeddings");

    TORCH_CHECK(input_features.dim() == 3, "input_features must be 3D, got ", input_features.dim(), "D");
    TORCH_CHECK(lm_embeddings.dim() == 3, "lm_embeddings must be 3D, got ", lm_embeddings.dim(), "D");
    TORCH_CHECK(audio_token_positions.dim() == 2, "audio_token_positions must be 2D, got ", audio_token_positions.dim(), "D");
    TORCH_CHECK(learnable_queries.dim() == 3, "learnable_queries must be 3D, got ", learnable_queries.dim(), "D");
    TORCH_CHECK(fused_embeddings.dim() == 3, "fused_embeddings must be 3D, got ", fused_embeddings.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(input_features.device());

    const int64_t batch_size = input_features.size(0);
    const int64_t num_audio_tokens = audio_token_positions.size(1);

    const int64_t encoder_hidden_dim = 512;
    const int64_t projector_hidden_size = 1024;
    const int64_t window_size = 15;
    const int64_t num_queries = 40;
    const int64_t qformer_num_heads = 16;
    const int64_t head_dim = projector_hidden_size / qformer_num_heads;

    // fp32 copies of every operand: the reference upcasts inside each linear.
    auto input_features_f = input_features.to(torch::kFloat32);
    auto encoder_input_weight_f = encoder_input_weight.to(torch::kFloat32);
    auto encoder_input_bias_f = encoder_input_bias.to(torch::kFloat32);
    auto encoder_out_weight_f = encoder_out_weight.to(torch::kFloat32);
    auto encoder_out_bias_f = encoder_out_bias.to(torch::kFloat32);
    auto encoder_out_mid_weight_f = encoder_out_mid_weight.to(torch::kFloat32);
    auto encoder_out_mid_bias_f = encoder_out_mid_bias.to(torch::kFloat32);
    auto qformer_q_proj_weight_f = qformer_q_proj_weight.to(torch::kFloat32);
    auto qformer_q_proj_bias_f = qformer_q_proj_bias.to(torch::kFloat32);
    auto qformer_k_proj_weight_f = qformer_k_proj_weight.to(torch::kFloat32);
    auto qformer_k_proj_bias_f = qformer_k_proj_bias.to(torch::kFloat32);
    auto qformer_v_proj_weight_f = qformer_v_proj_weight.to(torch::kFloat32);
    auto qformer_v_proj_bias_f = qformer_v_proj_bias.to(torch::kFloat32);
    auto qformer_out_proj_weight_f = qformer_out_proj_weight.to(torch::kFloat32);
    auto qformer_out_proj_bias_f = qformer_out_proj_bias.to(torch::kFloat32);
    auto projector_weight_f = projector_weight.to(torch::kFloat32);
    auto projector_bias_f = projector_bias.to(torch::kFloat32);

    // Stage 1: encoder input projection plus a simplified mid-layer CTC feedback.
    auto hidden_states =
        at::linear(input_features_f, encoder_input_weight_f, encoder_input_bias_f);
    auto hidden_states_mid =
        at::linear(hidden_states, encoder_out_weight_f, encoder_out_bias_f);
    auto softmax_out = at::softmax(hidden_states_mid, /*dim=*/-1);
    auto feedback =
        at::linear(softmax_out, encoder_out_mid_weight_f, encoder_out_mid_bias_f);
    auto encoder_output = hidden_states + feedback;  // [batch, audio_seq_len, 512]

    // Stage 2: windowed Q-Former downsampling.
    const int64_t seq_len = encoder_output.size(1);
    const int64_t nblocks =
        static_cast<int64_t>(std::ceil(static_cast<double>(seq_len) / window_size));
    const int64_t pad = nblocks * window_size - seq_len;

    auto encoder_output_padded = encoder_output;
    if (pad > 0) {
        // Pad only the sequence dim (second-to-last) by `pad` at the end.
        encoder_output_padded =
            at::constant_pad_nd(encoder_output, {0, 0, 0, pad}, 0);
    }

    const int64_t bsz_nblocks = batch_size * nblocks;
    auto encoder_output_windowed =
        encoder_output_padded.view({bsz_nblocks, window_size, encoder_hidden_dim});

    // Broadcast learnable queries to every window, upcast to fp32.
    auto queries = learnable_queries.expand({bsz_nblocks, -1, -1}).to(torch::kFloat32);

    auto Q = at::linear(queries, qformer_q_proj_weight_f, qformer_q_proj_bias_f);
    auto K = at::linear(encoder_output_windowed, qformer_k_proj_weight_f, qformer_k_proj_bias_f);
    auto V = at::linear(encoder_output_windowed, qformer_v_proj_weight_f, qformer_v_proj_bias_f);

    // Reshape for multi-head attention: [bsz_nblocks, heads, seq, head_dim].
    Q = Q.view({bsz_nblocks, num_queries, qformer_num_heads, head_dim}).transpose(1, 2);
    K = K.view({bsz_nblocks, window_size, qformer_num_heads, head_dim}).transpose(1, 2);
    V = V.view({bsz_nblocks, window_size, qformer_num_heads, head_dim}).transpose(1, 2);

    // Scaled dot-product attention.
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    auto attn_weights = at::matmul(Q, K.transpose(-2, -1)) * scale;
    attn_weights = at::softmax(attn_weights, /*dim=*/-1);
    auto attn_output = at::matmul(attn_weights, V);

    // Merge heads back: [bsz_nblocks, num_queries, projector_hidden_size].
    attn_output = attn_output.transpose(1, 2).contiguous()
                      .view({bsz_nblocks, num_queries, projector_hidden_size});

    // Q-Former output projection.
    auto query_output =
        at::linear(attn_output, qformer_out_proj_weight_f, qformer_out_proj_bias_f);

    // Back to batch dimension, then project to the LM hidden size.
    query_output =
        query_output.view({batch_size, nblocks * num_queries, projector_hidden_size});
    auto audio_embeddings =
        at::linear(query_output, projector_weight_f, projector_bias_f);
    // [batch, nblocks * num_queries, text_hidden_size]

    // Stage 3: scatter audio embeddings into the (fp32) LM embeddings per batch.
    auto fused = lm_embeddings.to(torch::kFloat32).clone();
    for (int64_t b = 0; b < batch_size; ++b) {
        auto positions = audio_token_positions[b];  // [num_audio_tokens]
        auto audio_emb = audio_embeddings[b].slice(0, 0, num_audio_tokens);
        fused[b].index_copy_(/*dim=*/0, positions, audio_emb);
    }

    auto result = fused.to(torch::kBFloat16);

    TORCH_CHECK(fused_embeddings.sizes() == result.sizes(),
                "fused_embeddings shape ", fused_embeddings.sizes(),
                " does not match computed ", result.sizes());
    fused_embeddings.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "Audio-to-language multimodal fusion. Pure ATen dispatch reproducing "
        "the reference: an fp32 CTC-style encoder (input projection plus a "
        "softmax-gated mid-layer feedback), an fp32 windowed Q-Former "
        "cross-attention downsampler with learnable queries (pad-to-window, "
        "multi-head scaled dot-product attention, output projection, projector "
        "to the LM hidden size), and a per-batch host loop that scatters the "
        "audio embeddings into an fp32 clone of the LM embeddings via "
        "index_copy_, with a single trailing bfloat16 cast on the fused result. "
        "Reproduced op-for-op from the reference. Destination-passing style "
        "(output written in place).");
}
