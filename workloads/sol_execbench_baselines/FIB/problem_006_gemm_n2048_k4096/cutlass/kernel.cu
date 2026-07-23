#include "kernel.h"

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>

// fp16 GEMM with fp32 accumulation: C = A @ B^T.
//
// PyTorch row-major layouts:
//   A is [M, K]  row-major  → describe as RowMajor with lda = K.
//   B is [N, K]  row-major  → B^T  is [K, N]; in column-major with
//                              leading dim K, element (k, i) lives at
//                              offset i*K + k, which is exactly where
//                              B[i, k] sits. So describe B as
//                              ColumnMajor with ldb = K.
//   C is [M, N]  row-major  → describe as RowMajor with ldc = N.
//
// Same Sm80 TensorOp template as the other GEMM landings. N=2048 is
// large enough to support several CTAs along the N axis; a 128x128
// threadblock tiles N evenly (16 tiles) and keeps M-flexible for the
// small-M workloads.

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

void problem_006_gemm_n2048_k4096_launcher(
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
