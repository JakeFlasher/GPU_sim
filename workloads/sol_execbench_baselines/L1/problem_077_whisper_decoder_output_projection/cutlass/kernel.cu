#include "kernel.h"

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/epilogue/thread/linear_combination.h>

// fp16 GEMM with fp32 accumulation: logits = hidden_states @ weight^T.
//
// Same row-major-to-mixed-layout encoding as the landed
// problem_004 / problem_007 / problem_008 / problem_009 /
// problem_010 / problem_011 GEMM cohort:
//   A (hidden_states): [M, K] row-major  -> RowMajor    with lda = K.
//   B (weight):        [N, K] row-major  -> ColumnMajor with ldb = K
//       (this is B^T col-major [K, N]).
//   C (logits):        [M, N] row-major  -> RowMajor    with ldc = N.
//
// Const axes pinned by the problem definition:
//   d_model    = K = 1280
//   vocab_size = N = 51866
//
// 51866 mod 8 = 2 -> the default `LinearCombination<half_t, 8, ...>`
// epilogue (AlignmentC = 8) would fail at kernel-init or produce
// out-of-bounds stores on the tail tile. Use `AlignmentC = 2`
// (LinearCombination kCount = 2) for a contract-faithful vector
// width that still satisfies 51866 mod 2 = 0.
//
// AlignmentA / AlignmentB stay at the default 8 (K = 1280 is
// 8-element aligned: 1280 mod 8 = 0).

namespace {

using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

constexpr int kAlignmentC = 2;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kAlignmentC,
    ElementAccumulator,
    ElementAccumulator>;

using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int kNumStages = 3;

using CutlassGemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    kNumStages>;

}  // namespace

void problem_077_whisper_decoder_output_projection_launcher(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& weight,
    torch::Tensor& logits_2d,
    cudaStream_t stream) {

    const int M = static_cast<int>(hidden_states_2d.size(0));
    const int K = static_cast<int>(hidden_states_2d.size(1));
    const int N = static_cast<int>(weight.size(0));

    auto const* A_ptr = reinterpret_cast<ElementInputA const*>(
        hidden_states_2d.data_ptr<at::Half>());
    auto const* B_ptr = reinterpret_cast<ElementInputB const*>(
        weight.data_ptr<at::Half>());
    auto* C_ptr = reinterpret_cast<ElementOutput*>(
        logits_2d.data_ptr<at::Half>());

    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args(
        {M, N, K},
        {A_ptr, K},
        {B_ptr, K},
        {C_ptr, N},
        {C_ptr, N},
        {1.0f, 0.0f});

    cutlass::Status status = gemm_operator(args, nullptr, stream);
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "CUTLASS GEMM (problem_077) failed: ",
        cutlassGetStatusString(status));
}
