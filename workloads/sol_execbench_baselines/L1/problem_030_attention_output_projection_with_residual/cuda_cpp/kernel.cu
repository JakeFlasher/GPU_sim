#include "kernel.h"

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>

// BF16 fused attention output projection + residual add:
//   output = (attn_output @ o_proj_weight.T) + residual
//
// Layout encoding:
//   attn_output: [B, S, H] row-major  →  flatten to [B*S, H];
//                describe as RowMajor with lda = H.
//   o_proj_weight: [H, H] row-major; we want `attn_output @ o_proj_weight.T`
//                which is equivalent to viewing o_proj_weight as
//                ColumnMajor with ldb = H. Same trick the fp16 GEMM
//                landings use to fuse the .T into the layout.
//   residual: [B, S, H] row-major  →  flatten to [B*S, H];
//             describe as RowMajor with ldc = H. Passed as the
//             GEMM's C source.
//   output: [B, S, H] row-major  →  flatten to [B*S, H];
//           describe as RowMajor with ldd = H. The GEMM's D
//           destination. Same buffer as the eval driver allocates
//           in DPS; the kernel writes to it in-place.
//
// The CUTLASS epilogue `LinearCombination` computes
//   D[i, j] = α · accumulator[i, j] + β · C[i, j]
// with α = 1.0 and β = 1.0, which is exactly the residual-add we
// need. No separate elementwise add kernel; the residual sectors
// stream through the epilogue alongside the GEMM output.
//
// We pick Sm80 TensorOp (same template family as the existing fp16
// GEMM landings) so the same code path compiles for both the local
// 3090 Ti (sm_86) and the Phase-2 B200 (sm_100a) target — Sm80 PTX
// is preserved by both. ThreadblockShape 128x128x32 / WarpShape
// 64x64x32 / InstructionShape 16x8x16 / NumStages 3 is the standard
// Ampere BF16 fp32-accumulator GEMM tile.

namespace {

using ElementInputA = cutlass::bfloat16_t;
using ElementInputB = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
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

void problem_030_attn_output_proj_residual_launcher(
    const torch::Tensor& attn_output,
    const torch::Tensor& residual,
    const torch::Tensor& o_proj_weight,
    torch::Tensor& output,
    cudaStream_t stream) {

    // Flatten [B, S, H] -> [B*S, H] without copying.
    const int M = static_cast<int>(attn_output.size(0) * attn_output.size(1));
    const int H = static_cast<int>(attn_output.size(2));
    // N = output hidden dim = H (o_proj_weight is square HxH).
    // K = input feature dim = H.
    const int N = H;
    const int K = H;

    auto const* A_ptr = reinterpret_cast<ElementInputA const*>(
        attn_output.data_ptr<at::BFloat16>());
    auto const* B_ptr = reinterpret_cast<ElementInputB const*>(
        o_proj_weight.data_ptr<at::BFloat16>());
    auto const* C_ptr = reinterpret_cast<ElementOutput const*>(
        residual.data_ptr<at::BFloat16>());
    auto* D_ptr = reinterpret_cast<ElementOutput*>(
        output.data_ptr<at::BFloat16>());

    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args(
        {M, N, K},
        {A_ptr, K},
        {B_ptr, K},
        {C_ptr, N},   // C source = residual
        {D_ptr, N},   // D destination = output
        {1.0f, 1.0f}); // alpha = 1, beta = 1 (residual-add)

    cutlass::Status status = gemm_operator(args, nullptr, stream);
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "CUTLASS GEMM failed: ", cutlassGetStatusString(status));
}
