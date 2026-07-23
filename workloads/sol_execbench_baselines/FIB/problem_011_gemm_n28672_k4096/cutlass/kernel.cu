#include "kernel.h"

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>

// fp16 GEMM with fp32 accumulation: C = A @ B^T.
//
// Same row-major-to-mixed-layout encoding as the landed problem_004 /
// problem_007 / problem_008 / problem_009 / problem_010 GEMM
// baselines:
//   A: [M, K] row-major  →  RowMajor    with lda = K.
//   B: [N, K] row-major  →  ColumnMajor with ldb = K (this is B^T col-major [K, N]).
//   C: [M, N] row-major  →  RowMajor    with ldc = N.
//
// N=28672 and K=4096 are pinned. The (28672, 4096) shape matches
// Llama 3.1 8B's mlp.gate_up_proj (the fused gate+up projection that
// produces the SiLU input pair). N=28672 is the largest N in the
// cohort B GEMM family; it tiles cleanly at 128 (28672 / 128 = 224
// N-tiles per row-stripe). The cohort partition marks this slug as
// `predictor_hostile` because the per-row B-stream working set is
// 28672 × 4096 × 2 = 224 MiB fp16 — far outside both the 3090 Ti's
// 6 MB L2 and B200's L2 partition, so the predictor must learn the
// cold-streaming behavior on a tile this large.

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

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator>;

using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 3;

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
    NumStages>;

}  // namespace

void problem_011_gemm_n28672_k4096_launcher(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    cudaStream_t stream) {

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));

    auto const* A_ptr =
        reinterpret_cast<ElementInputA const*>(A.data_ptr<at::Half>());
    auto const* B_ptr =
        reinterpret_cast<ElementInputB const*>(B.data_ptr<at::Half>());
    auto* C_ptr =
        reinterpret_cast<ElementOutput*>(C.data_ptr<at::Half>());

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
        "CUTLASS GEMM failed: ", cutlassGetStatusString(status));
}
