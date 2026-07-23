#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Backward pass for rotary position embedding with attention scaling.
//
// Reference (problem_014_rotary_embedding_with_attention_scaling_backward):
//     grad_cos = grad_cos.float()                       # bf16 -> fp32
//     grad_sin = grad_sin.float()                       # bf16 -> fp32
//     half_dim = head_dim // 2
//     grad_cos_freqs = grad_cos[..., :half_dim] + grad_cos[..., half_dim:]
//     grad_sin_freqs = grad_sin[..., :half_dim] + grad_sin[..., half_dim:]
//     emb_half = emb[..., :half_dim]
//     grad_emb = (grad_cos_freqs * (-emb_half.sin()) * attention_scaling
//                 + grad_sin_freqs *  emb_half.cos()  * attention_scaling)
//     grad_freqs = grad_emb.transpose(1, 2)             # (B, half_dim, seq_len)
//     grad_position_ids_expanded =
//         inv_freq_expanded.transpose(-2, -1) @ grad_freqs   # (B,1,H)@(B,H,S)->(B,1,S)
//     grad_position_ids = grad_position_ids_expanded.squeeze(1)   # (B, seq_len)
//
// Per definition.json (insertion order):
//   grad_cos           : [batch, seq_len, head_dim]   bf16
//   grad_sin           : [batch, seq_len, head_dim]   bf16
//   emb                : [batch, seq_len, head_dim]   fp32
//   inv_freq_expanded  : [batch, half_dim, 1]         fp32
//   attention_scaling  : scalar                       fp32   (shape: null ->
//                                                              a plain C++ double)
//   grad_position_ids  : [batch, seq_len]             fp32   (DPS, pre-alloc)
//
// Implementation: pure ATen dispatch -- NO custom CUDA kernels. The reference's
// tensor algebra is reproduced op-for-op. The bf16->fp32 cast on grad_cos /
// grad_sin happens up-front, exactly as the reference's .float() calls, before
// any arithmetic; all subsequent math is fp32. attention_scaling is a scalar
// (shape null in definition.json) and is applied to BOTH the -sin (cos-grad)
// and +cos (sin-grad) terms. Output is written in place into the pre-allocated
// destination (destination-passing style).

void run(
    const torch::Tensor& grad_cos,
    const torch::Tensor& grad_sin,
    const torch::Tensor& emb,
    const torch::Tensor& inv_freq_expanded,
    double attention_scaling,
    torch::Tensor grad_position_ids
) {
    TORCH_CHECK(grad_cos.is_cuda(), "grad_cos must be a CUDA tensor");
    TORCH_CHECK(grad_sin.is_cuda(), "grad_sin must be a CUDA tensor");
    TORCH_CHECK(emb.is_cuda(), "emb must be a CUDA tensor");
    TORCH_CHECK(inv_freq_expanded.is_cuda(), "inv_freq_expanded must be a CUDA tensor");
    TORCH_CHECK(grad_position_ids.is_cuda(), "grad_position_ids must be a CUDA tensor");

    TORCH_CHECK(grad_cos.dtype() == torch::kBFloat16, "grad_cos must be bfloat16, got ", grad_cos.dtype());
    TORCH_CHECK(grad_sin.dtype() == torch::kBFloat16, "grad_sin must be bfloat16, got ", grad_sin.dtype());
    TORCH_CHECK(emb.dtype() == torch::kFloat32, "emb must be float32, got ", emb.dtype());
    TORCH_CHECK(inv_freq_expanded.dtype() == torch::kFloat32, "inv_freq_expanded must be float32, got ", inv_freq_expanded.dtype());
    TORCH_CHECK(grad_position_ids.dtype() == torch::kFloat32, "grad_position_ids must be float32, got ", grad_position_ids.dtype());

    TORCH_CHECK(grad_cos.dim() == 3, "grad_cos must be 3D [batch, seq_len, head_dim], got ", grad_cos.dim(), "D");
    TORCH_CHECK(grad_sin.dim() == 3, "grad_sin must be 3D [batch, seq_len, head_dim], got ", grad_sin.dim(), "D");
    TORCH_CHECK(emb.dim() == 3, "emb must be 3D [batch, seq_len, head_dim], got ", emb.dim(), "D");
    TORCH_CHECK(inv_freq_expanded.dim() == 3, "inv_freq_expanded must be 3D [batch, half_dim, 1], got ", inv_freq_expanded.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_cos.device());

    const int64_t head_dim = emb.size(-1);
    const int64_t half_dim = head_dim / 2;

    // bf16 -> fp32 up-front, exactly as the reference's grad_cos.float() /
    // grad_sin.float(). All subsequent arithmetic is fp32.
    torch::Tensor grad_cos_f = grad_cos.to(torch::kFloat32);
    torch::Tensor grad_sin_f = grad_sin.to(torch::kFloat32);

    // Split each grad into its two head_dim halves and sum (the cat(freqs,freqs)
    // duplication in forward sends both halves' gradients back to freqs).
    torch::Tensor grad_cos_freqs =
        grad_cos_f.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim) +
        grad_cos_f.slice(/*dim=*/-1, /*start=*/half_dim, /*end=*/head_dim);
    torch::Tensor grad_sin_freqs =
        grad_sin_f.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim) +
        grad_sin_f.slice(/*dim=*/-1, /*start=*/half_dim, /*end=*/head_dim);

    // emb_half = emb[..., :half_dim].
    torch::Tensor emb_half = emb.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim);

    // grad_emb = grad_cos_freqs * (-sin(emb_half)) * scaling
    //          + grad_sin_freqs *   cos(emb_half)  * scaling
    torch::Tensor grad_emb =
        grad_cos_freqs * (-at::sin(emb_half)) * attention_scaling +
        grad_sin_freqs * at::cos(emb_half) * attention_scaling;

    // Transpose freqs back: (B, seq_len, half_dim) -> (B, half_dim, seq_len).
    torch::Tensor grad_freqs = grad_emb.transpose(1, 2);

    // Gradient through the matmul:
    //   inv_freq_expanded.T : (B, 1, half_dim)
    //   grad_freqs          : (B, half_dim, seq_len)
    //   -> (B, 1, seq_len)
    torch::Tensor grad_position_ids_expanded =
        at::matmul(inv_freq_expanded.transpose(-2, -1), grad_freqs);

    // Squeeze the singleton dim: (B, 1, seq_len) -> (B, seq_len).
    torch::Tensor result = grad_position_ids_expanded.squeeze(1);

    TORCH_CHECK(result.sizes() == grad_position_ids.sizes(),
                "computed grad_position_ids shape ", result.sizes(),
                " does not match pre-allocated output ", grad_position_ids.sizes());
    grad_position_ids.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "RoPE-with-attention-scaling backward: grad_position_ids via bf16->fp32 "
          "grad cast, half-split sum, scaled -sin/+cos grad, transpose, and "
          "batched matmul(inv_freq_expanded.T, grad_freqs); destination-passing "
          "style.");
}
