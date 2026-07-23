#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// Backward pass for vision rotary position embedding generation.
//
// Reference (problem_024_vision_rotary_position_embedding_generation_backward):
//     max_grid_size = pos_ids.max().item() + 1
//     grad_emb = -grad_cos * emb.sin() + grad_sin * emb.cos()
//     grad_rotary_pos_emb = grad_emb[:, :head_dim//2] + grad_emb[:, head_dim//2:]
//     grad_freqs_indexed = grad_rotary_pos_emb.reshape(total_patches, 2, head_dim_quarter)
//     grad_freqs = zeros(max_grid_size, head_dim_quarter, dtype=inv_freq.dtype)
//     grad_freqs.index_add_(0, pos_ids[:, 0], grad_freqs_indexed[:, 0, :])
//     grad_freqs.index_add_(0, pos_ids[:, 1], grad_freqs_indexed[:, 1, :])
//     seq = arange(max_grid_size, dtype=inv_freq.dtype)
//     grad_inv_freq = matmul(seq, grad_freqs)            # [Q]
//
// Per definition.json (insertion order):
//   grad_cos       : [total_patches, head_dim]      fp32
//   grad_sin       : [total_patches, head_dim]      fp32
//   pos_ids        : [total_patches, 2]             int64
//   inv_freq       : [head_dim_quarter]             fp32  (used only for its
//                                                          length / dtype / device,
//                                                          exactly as the reference)
//   emb            : [total_patches, head_dim]      fp32
//   grad_inv_freq  : [head_dim_quarter]             fp32  (DPS, pre-alloc)
//
// Implementation: pure ATen dispatch -- NO custom CUDA kernels. The reference's
// tensor algebra is reproduced op-for-op (sin/cos, split+sum of the cat,
// reshape, two index_add_ scatters, arange, matmul). The single host sync is
// the pos_ids.max().item() that sizes the scratch grad_freqs / arange, matching
// the reference's one .item(). Output is written in place into the pre-allocated
// destination (destination-passing style).

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& grad_cos,
    const torch::Tensor& grad_sin,
    const torch::Tensor& pos_ids,
    const torch::Tensor& inv_freq,
    const torch::Tensor& emb,
    torch::Tensor grad_inv_freq
) {
    check_fp32_cuda(grad_cos, "grad_cos");
    check_fp32_cuda(grad_sin, "grad_sin");
    check_fp32_cuda(inv_freq, "inv_freq");
    check_fp32_cuda(emb, "emb");
    check_fp32_cuda(grad_inv_freq, "grad_inv_freq");
    TORCH_CHECK(pos_ids.is_cuda(), "pos_ids must be a CUDA tensor");
    TORCH_CHECK(pos_ids.dtype() == torch::kInt64, "pos_ids must be int64, got ", pos_ids.dtype());

    TORCH_CHECK(grad_cos.dim() == 2, "grad_cos must be 2D [total_patches, head_dim], got ", grad_cos.dim(), "D");
    TORCH_CHECK(grad_sin.dim() == 2, "grad_sin must be 2D [total_patches, head_dim], got ", grad_sin.dim(), "D");
    TORCH_CHECK(emb.dim() == 2, "emb must be 2D [total_patches, head_dim], got ", emb.dim(), "D");
    TORCH_CHECK(pos_ids.dim() == 2, "pos_ids must be 2D [total_patches, 2], got ", pos_ids.dim(), "D");
    TORCH_CHECK(pos_ids.size(1) == 2, "pos_ids must have 2 columns (h, w), got ", pos_ids.size(1));
    TORCH_CHECK(inv_freq.dim() == 1, "inv_freq must be 1D [head_dim_quarter], got ", inv_freq.dim(), "D");

    const c10::cuda::CUDAGuard device_guard(grad_cos.device());

    const int64_t total_patches = grad_cos.size(0);
    const int64_t head_dim = grad_cos.size(1);
    const int64_t head_dim_half = head_dim / 2;
    const int64_t head_dim_quarter = inv_freq.size(0);

    // Stage 1: gradient through cos / sin.
    //   grad_emb = -grad_cos * emb.sin() + grad_sin * emb.cos()
    torch::Tensor grad_emb = -grad_cos * at::sin(emb) + grad_sin * at::cos(emb);

    // Stage 2: gradient through the cat((rotary, rotary), dim=-1) -> split + sum.
    //   grad_rotary_pos_emb = grad_emb[:, :head_dim//2] + grad_emb[:, head_dim//2:]
    torch::Tensor grad_rotary_pos_emb =
        grad_emb.slice(/*dim=*/1, /*start=*/0, /*end=*/head_dim_half) +
        grad_emb.slice(/*dim=*/1, /*start=*/head_dim_half, /*end=*/head_dim);

    // Stage 3: gradient through flatten -> unflatten to (total_patches, 2, Q).
    torch::Tensor grad_freqs_indexed =
        grad_rotary_pos_emb.reshape({total_patches, 2, head_dim_quarter});

    // Stage 4: gradient through indexing -> scatter via index_add_ into the
    // full freqs table. The table is sized by the largest position id, which is
    // the reference's single .item() host sync.
    const int64_t max_grid_size = pos_ids.max().item<int64_t>() + 1;

    torch::Tensor grad_freqs = at::zeros(
        {max_grid_size, head_dim_quarter},
        inv_freq.options());

    // Scatter height-position gradients (column 0), then width (column 1).
    grad_freqs.index_add_(
        /*dim=*/0,
        /*index=*/pos_ids.select(/*dim=*/1, /*index=*/0),
        /*source=*/grad_freqs_indexed.select(/*dim=*/1, /*index=*/0));
    grad_freqs.index_add_(
        /*dim=*/0,
        /*index=*/pos_ids.select(/*dim=*/1, /*index=*/1),
        /*source=*/grad_freqs_indexed.select(/*dim=*/1, /*index=*/1));

    // Stage 5: gradient through outer(seq, inv_freq) -> matmul(seq, grad_freqs).
    //   seq : [max_grid_size]        grad_freqs : [max_grid_size, Q]  -> [Q]
    torch::Tensor seq = at::arange(max_grid_size, inv_freq.options());
    torch::Tensor result = at::matmul(seq, grad_freqs);

    TORCH_CHECK(result.sizes() == grad_inv_freq.sizes(),
                "computed grad_inv_freq shape ", result.sizes(),
                " does not match pre-allocated output ", grad_inv_freq.sizes());
    grad_inv_freq.copy_(result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "Vision RoPE generation backward: grad_inv_freq via sin/cos grad, "
          "split+sum, index_add_ scatter, and matmul(seq, grad_freqs); fp32; "
          "destination-passing style.");
}
