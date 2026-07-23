#include "kernel.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/epilogue/thread/linear_combination.h>

// Seven-kernel chain for the Vision MLP + GELU-tanh backward
// pass:
//
//   1. GEMM-A  (CUTLASS fp32 tf32-mma):
//        grad_gelu_output = grad_output @ fc2_weight
//        M = S, N = intermediate_size = 3584, K = hidden_size = 1280.
//
//   2. GEMM-B  (CUTLASS fp32 tf32-mma):
//        grad_fc2_weight = grad_output.t() @ gelu_output
//        M = hidden_size = 1280, N = intermediate_size = 3584, K = S.
//
//   3. Bias-sum kernel: grad_fc2_bias[col] = sum_{i in [0..S)} grad_output[i, col].
//        Grid (hidden_size = 1280); BLOCK_THREADS = 256; stride-loop
//        over S, warp-shuffle + 8-way cross-warp tree, lane 0 writes.
//
//   4. GELU-tanh backward per-element kernel:
//        for each (i, j) in [S, intermediate_size]:
//          x       = fc1_output[i, j]
//          tanh_a  = tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
//          sech_sq = 1 - tanh_a * tanh_a
//          d_arg   = sqrt(2/pi) * (1 + 3*0.044715 * x^2)
//          gelu_grad = 0.5*(1 + tanh_a) + 0.5*x*sech_sq*d_arg
//          grad_fc1_output[i, j] = grad_gelu_output[i, j] * gelu_grad
//
//   5. GEMM-C  (CUTLASS fp32 tf32-mma):
//        grad_hidden_state = grad_fc1_output @ fc1_weight
//        M = S, N = hidden_size = 1280, K = intermediate_size = 3584.
//
//   6. GEMM-D  (CUTLASS fp32 tf32-mma):
//        grad_fc1_weight = grad_fc1_output.t() @ hidden_state
//        M = intermediate_size = 3584, N = hidden_size = 1280, K = S.
//
//   7. Bias-sum kernel: grad_fc1_bias[col] = sum_{i in [0..S)} grad_fc1_output[i, col].
//        Grid (intermediate_size = 3584); BLOCK_THREADS = 256.
//
// All per-column bias-sum reductions are in-CTA only; no cross-CTA
// reduction tree avoids accumulation-order drift against PyTorch's
// reduction order. fp32 throughout matches the workload tolerance
// the canonical workloads use.

namespace {

constexpr int kHiddenSize = 1280;
constexpr int kIntermediateSize = 3584;
constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockThreads / kWarpSize;

// CUTLASS Sm80 TensorOp fp32 GEMM template.
using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;
using ElementAccumulator = float;

using LayoutRowMajor = cutlass::layout::RowMajor;
using LayoutColMajor = cutlass::layout::ColumnMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // = 4
    ElementAccumulator,
    ElementAccumulator>;

using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int kNumStages = 3;

// Two GEMM template instantiations differ only in B's layout: GEMMs
// that compute `A @ B` use B as RowMajor (B's logical shape is
// [K, N], stored row-major); GEMMs that compute `A.t() @ B` use the
// A operand as ColumnMajor over an [M, K] view of an underlying
// [K, M] row-major tensor.
//
// Concrete mapping:
//   GEMM-A: grad_output[S, K=1280] @ fc2_weight[K=1280, N=3584].
//     fc2_weight is stored as [hidden_size, intermediate_size]
//     row-major -> LayoutInputB = RowMajor.
//   GEMM-C: grad_fc1_output[S, K=3584] @ fc1_weight[K=3584, N=1280].
//     fc1_weight is stored as [intermediate_size, hidden_size]
//     row-major. Logical [K=3584, N=1280] equals the storage layout
//     -> LayoutInputB = RowMajor.
using GemmAxC = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutRowMajor,
    ElementInputB, LayoutRowMajor,
    ElementOutput, LayoutRowMajor,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    kNumStages>;

// GEMM-B and GEMM-D both compute `lhs.t() @ rhs`. The transposed-
// lhs case is encoded by giving A a ColumnMajor layout over its
// underlying row-major storage: a [K, M] row-major tensor read with
// ColumnMajor layout is the same as the transposed [M, K] view.
//   GEMM-B: grad_output[S, hidden_size=1280] (RowMajor) reinterpreted
//     as ColumnMajor -> [M=hidden_size, K=S]; gelu_output[S, N=3584]
//     RowMajor -> [K=S, N=3584].
//   GEMM-D: grad_fc1_output[S, intermediate_size=3584] (RowMajor)
//     reinterpreted as ColumnMajor -> [M=intermediate_size, K=S];
//     hidden_state[S, N=1280] RowMajor -> [K=S, N=1280].
using GemmBxD = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutColMajor,
    ElementInputB, LayoutRowMajor,
    ElementOutput, LayoutRowMajor,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    kNumStages>;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(
    float v, float* warp_sums
) {
    v = warp_reduce_sum(v);
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    if (lane == 0) warp_sums[warp_id] = v;
    __syncthreads();
    float total = 0.0f;
    if (threadIdx.x < kNumWarps) {
        total = warp_sums[threadIdx.x];
    }
    total = warp_reduce_sum(total);
    return total;
}

// Bias-sum kernel: one CTA per output column; stride-loop over S.
// Grid (num_cols); BLOCK_THREADS = 256.
__global__ void __launch_bounds__(kBlockThreads, 4) k_bias_column_sum(
    const float* __restrict__ x,
    float* __restrict__ bias_out,
    int S, int num_cols
) {
    const int col = blockIdx.x;
    if (col >= num_cols) return;
    const int tid = threadIdx.x;

    float partial = 0.0f;
    for (int i = tid; i < S; i += kBlockThreads) {
        partial += x[(size_t)i * num_cols + col];
    }

    __shared__ float warp_sums[kNumWarps];
    float total = block_reduce_sum(partial, warp_sums);
    if (tid == 0) {
        bias_out[col] = total;
    }
}

// GELU-tanh backward per-element kernel.
//   grad_fc1_output[i, j] = grad_gelu_output[i, j] *
//       (0.5*(1+tanh(arg)) + 0.5*x*sech^2(arg)*d_arg_dx)
//   arg       = sqrt(2/pi) * (x + 0.044715 * x^3)
//   d_arg_dx  = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
__global__ void __launch_bounds__(kBlockThreads, 4) k_gelu_tanh_backward(
    const float* __restrict__ grad_gelu_output,
    const float* __restrict__ fc1_output,
    float* __restrict__ grad_fc1_output,
    size_t total_elements
) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float c = 0.044715f;

    const size_t idx_base =
        (size_t)blockIdx.x * kBlockThreads + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * kBlockThreads;
    for (size_t idx = idx_base; idx < total_elements; idx += stride) {
        const float x = fc1_output[idx];
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float arg = sqrt_2_over_pi * (x + c * x3);
        const float t = tanhf(arg);
        const float sech_sq = 1.0f - t * t;
        const float d_arg_dx = sqrt_2_over_pi * (1.0f + 3.0f * c * x2);
        const float gelu_grad =
            0.5f * (1.0f + t) + 0.5f * x * sech_sq * d_arg_dx;
        grad_fc1_output[idx] = grad_gelu_output[idx] * gelu_grad;
    }
}

}  // namespace

void problem_022_vision_mlp_gelu_backward_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& hidden_state,
    const torch::Tensor& fc1_weight,
    const torch::Tensor& fc2_weight,
    const torch::Tensor& fc1_output,
    const torch::Tensor& gelu_output,
    torch::Tensor& grad_hidden_state,
    torch::Tensor& grad_fc1_weight,
    torch::Tensor& grad_fc1_bias,
    torch::Tensor& grad_fc2_weight,
    torch::Tensor& grad_fc2_bias,
    cudaStream_t stream
) {
    const int S = static_cast<int>(grad_output.size(0));

    auto opts = grad_output.options();
    auto grad_gelu_output =
        torch::empty({S, kIntermediateSize}, opts);
    auto grad_fc1_output = torch::empty({S, kIntermediateSize}, opts);

    // GEMM-A: grad_gelu_output = grad_output @ fc2_weight.
    //   grad_output  : [S, K=1280] RowMajor.
    //   fc2_weight   : [K=1280, N=3584] RowMajor.
    {
        GemmAxC gemm_op;
        typename GemmAxC::Arguments args(
            {S, kIntermediateSize, kHiddenSize},
            {grad_output.data_ptr<float>(), kHiddenSize},
            {fc2_weight.data_ptr<float>(), kIntermediateSize},
            {grad_gelu_output.data_ptr<float>(), kIntermediateSize},
            {grad_gelu_output.data_ptr<float>(), kIntermediateSize},
            {1.0f, 0.0f});
        cutlass::Status s = gemm_op(args, nullptr, stream);
        TORCH_CHECK(s == cutlass::Status::kSuccess,
                    "problem_022 GEMM-A failed: ",
                    cutlass::cutlassGetStatusString(s));
    }

    // GEMM-B: grad_fc2_weight = grad_output.t() @ gelu_output.
    //   grad_output  : underlying RowMajor [S, hidden_size=1280];
    //     reinterpreted as ColumnMajor -> [M=hidden_size, K=S],
    //     leading dim = hidden_size.
    //   gelu_output  : [K=S, N=3584] RowMajor.
    {
        GemmBxD gemm_op;
        typename GemmBxD::Arguments args(
            {kHiddenSize, kIntermediateSize, S},
            {grad_output.data_ptr<float>(), kHiddenSize},
            {gelu_output.data_ptr<float>(), kIntermediateSize},
            {grad_fc2_weight.data_ptr<float>(), kIntermediateSize},
            {grad_fc2_weight.data_ptr<float>(), kIntermediateSize},
            {1.0f, 0.0f});
        cutlass::Status s = gemm_op(args, nullptr, stream);
        TORCH_CHECK(s == cutlass::Status::kSuccess,
                    "problem_022 GEMM-B failed: ",
                    cutlass::cutlassGetStatusString(s));
    }

    // grad_fc2_bias = grad_output.sum(dim=0).
    {
        dim3 grid(kHiddenSize);
        dim3 block(kBlockThreads);
        k_bias_column_sum<<<grid, block, 0, stream>>>(
            grad_output.data_ptr<float>(),
            grad_fc2_bias.data_ptr<float>(),
            S, kHiddenSize);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "problem_022 grad_fc2_bias kernel launch failed");
    }

    // GELU-tanh backward: writes grad_fc1_output.
    {
        const size_t total = (size_t)S * kIntermediateSize;
        const int max_blocks = 65535;
        int blocks = static_cast<int>(
            (total + kBlockThreads - 1) / kBlockThreads);
        if (blocks > max_blocks) blocks = max_blocks;
        dim3 grid(blocks);
        dim3 block(kBlockThreads);
        k_gelu_tanh_backward<<<grid, block, 0, stream>>>(
            grad_gelu_output.data_ptr<float>(),
            fc1_output.data_ptr<float>(),
            grad_fc1_output.data_ptr<float>(),
            total);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "problem_022 GELU backward kernel launch failed");
    }

    // GEMM-C: grad_hidden_state = grad_fc1_output @ fc1_weight.
    //   grad_fc1_output : [S, K=3584] RowMajor.
    //   fc1_weight      : [K=3584, N=1280] RowMajor (stored as
    //                     [intermediate_size, hidden_size]).
    {
        GemmAxC gemm_op;
        typename GemmAxC::Arguments args(
            {S, kHiddenSize, kIntermediateSize},
            {grad_fc1_output.data_ptr<float>(), kIntermediateSize},
            {fc1_weight.data_ptr<float>(), kHiddenSize},
            {grad_hidden_state.data_ptr<float>(), kHiddenSize},
            {grad_hidden_state.data_ptr<float>(), kHiddenSize},
            {1.0f, 0.0f});
        cutlass::Status s = gemm_op(args, nullptr, stream);
        TORCH_CHECK(s == cutlass::Status::kSuccess,
                    "problem_022 GEMM-C failed: ",
                    cutlass::cutlassGetStatusString(s));
    }

    // GEMM-D: grad_fc1_weight = grad_fc1_output.t() @ hidden_state.
    //   grad_fc1_output : underlying RowMajor [S, intermediate_size];
    //     reinterpreted as ColumnMajor -> [M=intermediate_size,
    //     K=S], leading dim = intermediate_size.
    //   hidden_state    : [K=S, N=1280] RowMajor.
    {
        GemmBxD gemm_op;
        typename GemmBxD::Arguments args(
            {kIntermediateSize, kHiddenSize, S},
            {grad_fc1_output.data_ptr<float>(), kIntermediateSize},
            {hidden_state.data_ptr<float>(), kHiddenSize},
            {grad_fc1_weight.data_ptr<float>(), kHiddenSize},
            {grad_fc1_weight.data_ptr<float>(), kHiddenSize},
            {1.0f, 0.0f});
        cutlass::Status s = gemm_op(args, nullptr, stream);
        TORCH_CHECK(s == cutlass::Status::kSuccess,
                    "problem_022 GEMM-D failed: ",
                    cutlass::cutlassGetStatusString(s));
    }

    // grad_fc1_bias = grad_fc1_output.sum(dim=0).
    {
        dim3 grid(kIntermediateSize);
        dim3 block(kBlockThreads);
        k_bias_column_sum<<<grid, block, 0, stream>>>(
            grad_fc1_output.data_ptr<float>(),
            grad_fc1_bias.data_ptr<float>(),
            S, kIntermediateSize);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "problem_022 grad_fc1_bias kernel launch failed");
    }
}
