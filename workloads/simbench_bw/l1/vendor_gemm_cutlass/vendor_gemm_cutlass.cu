// Project-owned CUTLASS successor: BF16 input GEMM with FP32 output.
//
// This translation unit deliberately contains no cuBLAS/cuBLASLt dependency.
// CUTLASS's SIMT device::Gemm is selected for the sm_86 witness because the
// generic tensor-op alias does not provide a valid default configuration for
// every BF16 layout on this host. The source, CUTLASS tree, compiler, target
// architecture, and resulting binary are pinned by the successor witness.

#include <cuda_runtime.h>

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

using Element = cutlass::bfloat16_t;
using ElementOutput = float;
using Gemm = cutlass::gemm::device::Gemm<
    Element,
    cutlass::layout::RowMajor,
    Element,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80>;

extern "C" int vendor_gemm_cutlass_launch(
    const Element *d_a,
    const Element *d_b_column_major,
    ElementOutput *d_c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
    Gemm gemm;
    Gemm::Arguments arguments(
        {m, n, k},
        {d_a, k},
        {d_b_column_major, k},
        {d_c, n},
        {d_c, n},
        {1.0f, 0.0f});

    cutlass::Status status = gemm.initialize(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return 1;
    }
    status = gemm();
    return status == cutlass::Status::kSuccess ? 0 : 2;
}
