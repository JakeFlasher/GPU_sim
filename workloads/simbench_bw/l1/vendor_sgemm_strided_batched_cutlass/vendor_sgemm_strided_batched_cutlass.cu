// Project-owned CUTLASS successor: FP32 strided-batched GEMM.
// This translation unit deliberately contains no cuBLAS/cuBLASLt dependency.

#include <cuda_runtime.h>

#include <cstdint>

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>

using Element = float;
using ElementOutput = float;
using Gemm = cutlass::gemm::device::GemmBatched<
    Element,
    cutlass::layout::RowMajor,
    Element,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80>;

extern "C" int vendor_sgemm_strided_batched_cutlass_launch(
    const Element *d_a,
    const Element *d_b_column_major,
    ElementOutput *d_c,
    int m,
    int n,
    int k,
    int batch_count,
    cudaStream_t stream) {
    Gemm gemm;
    const int64_t stride_a = static_cast<int64_t>(m) * k;
    const int64_t stride_b = static_cast<int64_t>(k) * n;
    const int64_t stride_c = static_cast<int64_t>(m) * n;
    Gemm::Arguments arguments(
        {m, n, k},
        {d_a, k}, stride_a,
        {d_b_column_major, k}, stride_b,
        {d_c, n}, stride_c,
        {d_c, n}, stride_c,
        {1.0f, 0.0f}, batch_count);

    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return 1;
    }
    status = gemm.initialize(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return 2;
    }
    status = gemm.run(stream);
    return status == cutlass::Status::kSuccess ? 0 : 3;
}
