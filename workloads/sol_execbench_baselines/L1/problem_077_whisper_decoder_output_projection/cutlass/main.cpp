#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

// Destination-passing-style entry for the Whisper decoder output
// projection. Per definition.json:
//   hidden_states : [batch_size, seq_len, d_model=1280] fp16
//   weight        : [vocab_size=51866, d_model=1280]    fp16
//   logits        : [batch_size, seq_len, vocab_size=51866] fp16 (DPS)
//
// Const axes pinned by the problem definition AND enforced on every
// tensor the kernel indexes with that fixed assumption, following
// the runtime-check audit discipline:
//   d_model    = 1280   pinned on hidden_states and weight.
//   vocab_size = 51866  pinned on weight and logits.
//
// The kernel reduces the 3D `[B, S, *]` tensors to flat 2D
// `[B*S, *]` views via `flatten(0, 1)` before calling the launcher;
// the launcher emits a single CUTLASS Sm80 TensorOp GEMM with
// `M = B*S`, `K = d_model = 1280`, `N = vocab_size = 51866`,
// `AlignmentC = 2` (since 51866 mod 8 = 2).

void run(
    torch::Tensor hidden_states,
    torch::Tensor weight,
    torch::Tensor logits
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");

    TORCH_CHECK(hidden_states.dtype() == torch::kFloat16,
                "hidden_states must be float16");
    TORCH_CHECK(weight.dtype() == torch::kFloat16,
                "weight must be float16");
    TORCH_CHECK(logits.dtype() == torch::kFloat16,
                "logits must be float16");

    TORCH_CHECK(hidden_states.is_contiguous(),
                "hidden_states must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");

    TORCH_CHECK(hidden_states.dim() == 3,
                "hidden_states must be 3D [B, S, d_model]");
    TORCH_CHECK(weight.dim() == 2,
                "weight must be 2D [vocab_size, d_model]");
    TORCH_CHECK(logits.dim() == 3,
                "logits must be 3D [B, S, vocab_size]");

    constexpr int64_t kDModel = 1280;
    constexpr int64_t kVocabSize = 51866;

    TORCH_CHECK(hidden_states.size(2) == kDModel,
                "hidden_states.size(2) must equal d_model=1280, got ",
                hidden_states.size(2));
    TORCH_CHECK(weight.size(1) == kDModel,
                "weight.size(1) must equal d_model=1280, got ",
                weight.size(1));
    TORCH_CHECK(weight.size(0) == kVocabSize,
                "weight.size(0) must equal vocab_size=51866, got ",
                weight.size(0));
    TORCH_CHECK(logits.size(2) == kVocabSize,
                "logits.size(2) must equal vocab_size=51866, got ",
                logits.size(2));
    TORCH_CHECK(logits.size(0) == hidden_states.size(0),
                "logits.size(0) must equal hidden_states.size(0)");
    TORCH_CHECK(logits.size(1) == hidden_states.size(1),
                "logits.size(1) must equal hidden_states.size(1)");

    auto hidden_states_2d = hidden_states.flatten(0, 1);
    auto logits_2d = logits.flatten(0, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    problem_077_whisper_decoder_output_projection_launcher(
        hidden_states_2d, weight, logits_2d, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run", &run,
        "fp16 GEMM logits = hidden_states @ weight.T via CUTLASS "
        "TensorOp on Sm80; destination-passing style. d_model=1280, "
        "vocab_size=51866 pinned; AlignmentC=2 epilogue.");
}
