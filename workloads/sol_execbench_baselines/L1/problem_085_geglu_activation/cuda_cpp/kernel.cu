#include "kernel.h"
#include <cuda_runtime.h>

// FP32 GEGLU (Gated GLU with GELU):
//   output[..., c] = GELU_tanh(x[..., c]) * x[..., inner_dim + c]
//
// GELU tanh approximation:
//   GELU_tanh(v) = 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 * v^3)))
//
// Layout: input x is [B, S, 2*inner_dim] row-major; the gate and
// linear halves are contiguous regions of each row separated by
// `inner_dim` fp32 elements. Output is [B, S, inner_dim] row-major.
//
// Vectorization: inner_dim is pinned at 5120, which is a multiple of
// 4, and ATen-allocated CUDA tensors are 16-byte aligned, so float4
// loads/stores aligned at column offsets of 4 within each row are
// safe for both the gate half (offset 0 within the row) and the
// linear half (offset inner_dim within the row). Each thread handles
// 1 float4 of output (4 consecutive cols within the same row),
// reading 1 float4 from the gate half and 1 float4 from the linear
// half at the matching cols.
//
// Numerical contract: the reference uses `torch.nn.functional.gelu(
// x_gate, approximate='tanh')` (FP32). `tanhf` matches the
// reference's tolerance.

namespace {

constexpr int kInnerDim = 5120;
constexpr int kInputCols = 2 * kInnerDim;
constexpr int kInnerDimVec4 = kInnerDim / 4;     // 1280
constexpr int kInputColsVec4 = kInputCols / 4;   // 2560

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kCoeff = 0.044715f;

__device__ __forceinline__ float gelu_tanh_approx_f(float x) {
    float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Each thread emits one float4 of output. There are
// `n_rows * kInnerDimVec4` total output float4s; the thread maps to
// `(row, col_v4)` via integer division / mod on its global thread
// index.
__global__ void problem_085_geglu_vec4_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ output,
    int n_rows) {

    const int idx_v4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_v4 = n_rows * kInnerDimVec4;
    if (idx_v4 >= total_v4) {
        return;
    }

    const int row = idx_v4 / kInnerDimVec4;
    const int col_v4 = idx_v4 - row * kInnerDimVec4;

    const int gate_v4   = row * kInputColsVec4 + col_v4;
    const int linear_v4 = gate_v4 + kInnerDimVec4;

    const float4 g = x[gate_v4];
    const float4 l = x[linear_v4];

    float4 out;
    out.x = gelu_tanh_approx_f(g.x) * l.x;
    out.y = gelu_tanh_approx_f(g.y) * l.y;
    out.z = gelu_tanh_approx_f(g.z) * l.z;
    out.w = gelu_tanh_approx_f(g.w) * l.w;
    output[idx_v4] = out;
}

}  // namespace

void problem_085_geglu_activation_launcher(
    const torch::Tensor& x,
    torch::Tensor& output,
    cudaStream_t stream) {

    // x is [B, S, 2*inner_dim] fp32; n_rows = B*S.
    const int n_rows = static_cast<int>(x.numel() / kInputCols);
    const int total_v4 = n_rows * kInnerDimVec4;

    constexpr int BLOCK_THREADS = 256;
    if (total_v4 == 0) {
        return;
    }
    const int grid = (total_v4 + BLOCK_THREADS - 1) / BLOCK_THREADS;
    problem_085_geglu_vec4_kernel<<<grid, BLOCK_THREADS, 0, stream>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        n_rows);
}
