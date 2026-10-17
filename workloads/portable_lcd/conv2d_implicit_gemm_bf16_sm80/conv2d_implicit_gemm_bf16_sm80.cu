// CUTLASS Conv2dFprop sm_80 implicit GEMM BF16 kernel.
//
// Thin wrapper around the vendored CUTLASS Conv2dFprop sm_80 entry
// templates. The vendored CUTLASS subset
// (workloads/portable_lcd/cutlass/upstream/) supplies the kernel
// templates verbatim from CUTLASS v3.8.0; this file instantiates
// only the Sm80 path (BF16 tensor-op, NHWC layouts, multistage
// pipeline, kOptimized iterator) and exposes a C-linkage launch
// entry consumed by main.cu.
//
// LCD discipline: only the Sm80 templates are reached. The aggregate
// CuTe headers transitively include sm_90 GMMA / sm_100 / sm_100 TMA
// trait declarations, but those are dead-stripped at compile time
// because we build with `-arch=compute_80` PTX plus base-real arch
// `sm_86 / sm_90 / sm_120` gencode flags. The fatbin-slice verifier
// and the suite SASS-coverage gate independently confirm no
// forbidden mnemonic (UTC*MMA / LDT / STT) leaks into the compiled
// sm_120 SASS.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace {

using ElementInputA      = cutlass::bfloat16_t;
using ElementInputB      = cutlass::bfloat16_t;
using ElementOutput      = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute     = float;

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape        = cutlass::gemm::GemmShape<32, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    /*ElementsPerAccess=*/8,
    ElementAccumulator,
    ElementCompute>;

using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

constexpr int kStages = 3;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    kStages,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using ImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

}  // namespace

extern "C" int conv2d_implicit_gemm_bf16_sm80_launch(
    const __nv_bfloat16 *d_act,    // [N, H, W, C]
    const __nv_bfloat16 *d_filter, // [K, R, S, C]
    __nv_bfloat16 *d_out,          // [N, P, Q, K]
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int P, int Q,
    cudaStream_t stream)
{
    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H, W, C},
        {K, R, S, C},
        {pad_h, pad_w, pad_h, pad_w},
        {stride_h, stride_w},
        {dilation_h, dilation_w},
        {N, P, Q, K},
        cutlass::conv::Mode::kCrossCorrelation,
        /*split_k_slices=*/1
    );

    // CUTLASS Conv2dFprop's device-class TensorRefA / TensorRefB use
    // non-const element types in their typedef; cast away const on
    // the input pointers to match. The kernel does not write to these
    // buffers.
    cutlass::TensorRef<ElementInputA, LayoutInputA> ref_a(
        const_cast<ElementInputA *>(reinterpret_cast<const ElementInputA *>(d_act)),
        LayoutInputA::packed({N, H, W, C}));
    cutlass::TensorRef<ElementInputB, LayoutInputB> ref_b(
        const_cast<ElementInputB *>(reinterpret_cast<const ElementInputB *>(d_filter)),
        LayoutInputB::packed({K, R, S, C}));
    cutlass::TensorRef<ElementOutput, LayoutOutput> ref_c(
        reinterpret_cast<ElementOutput *>(d_out),
        LayoutOutput::packed({N, P, Q, K}));
    cutlass::TensorRef<ElementOutput, LayoutOutput> ref_d = ref_c;

    typename ImplicitGemm::Arguments args{
        problem_size,
        ref_a, ref_b, ref_c, ref_d,
        { ElementCompute(1.0f), ElementCompute(0.0f) }
    };

    ImplicitGemm op;

    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "conv2d_can_implement_failed: %d\n",
                     static_cast<int>(status));
        return 1;
    }

    size_t workspace_bytes = op.get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_bytes > 0) {
        if (cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) {
            std::fprintf(stderr, "conv2d_workspace_alloc_failed: %zu bytes\n",
                         workspace_bytes);
            return 2;
        }
    }

    status = op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "conv2d_initialize_failed: %d\n",
                     static_cast<int>(status));
        if (workspace) cudaFree(workspace);
        return 3;
    }

    status = op(stream);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "conv2d_run_failed: %d\n",
                     static_cast<int>(status));
        if (workspace) cudaFree(workspace);
        return 4;
    }

    if (workspace) cudaFree(workspace);
    return 0;
}
