#include "kernel.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/epilogue/thread/linear_combination.h>

// Four-kernel chain for the FLUX feedforward (fp32):
//
//   GEMM 1 : workspace[B*S, 12288] = hidden_states[B*S, 3072]
//                                     @ fc1_weight^T[3072, 12288]
//            (CUTLASS Sm80 TensorOp tf32 mma; same row-major-to-
//            mixed-layout encoding as the landed problem_004 /
//            problem_007 / problem_011 GEMM cohort.)
//
//   ACT    : per-(b, s, n) thread reads workspace[bs, n], adds
//            fc1_bias[n], applies GELU-tanh approximation, writes
//            back to workspace[bs, n].
//
//   GEMM 2 : output[B*S, 3072] = workspace[B*S, 12288]
//                                @ fc2_weight^T[12288, 3072]
//            (same CUTLASS template).
//
//   BIAS   : per-(b, s, n) thread reads output[bs, n], adds
//            fc2_bias[n], writes back.
//
// Const axes pinned by the problem definition:
//   hidden_dim     = 3072
//   mlp_hidden_dim = 12288

namespace {

constexpr int kHiddenDim = 3072;
constexpr int kMlpHiddenDim = 12288;
constexpr int kBlockThreads = 256;

using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;
using ElementAccumulator = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

// fp32 Sm80 TensorOp uses mma.m16n8k8 tf32 mma internally (matches
// cuBLAS sgemm's tf32 default on Ampere). ThreadblockShape tiles
// 128x128 over M/N with 16 elements per tile-K column (fp32 fits
// fewer K elements per stage than fp16/bf16).
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

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluTanhConst = 0.044715f;

__device__ __forceinline__ float gelu_tanh_approx(float z) {
    const float inner = kSqrt2OverPi * (z + kGeluTanhConst * z * z * z);
    return 0.5f * z * (1.0f + tanhf(inner));
}

__global__ void __launch_bounds__(kBlockThreads, 8)
    k_gelu_tanh_with_bias_inplace(
        float* __restrict__ workspace,
        const float* __restrict__ bias,
        int total_M
) {
    const size_t total_elements =
        (size_t)total_M * (size_t)kMlpHiddenDim;
    const size_t stride = (size_t)gridDim.x * (size_t)kBlockThreads;
    size_t i = (size_t)blockIdx.x * (size_t)kBlockThreads
               + (size_t)threadIdx.x;
    for (; i < total_elements; i += stride) {
        const int n = (int)(i % (size_t)kMlpHiddenDim);
        const float val = workspace[i] + bias[n];
        workspace[i] = gelu_tanh_approx(val);
    }
}

__global__ void __launch_bounds__(kBlockThreads, 8)
    k_bias_add_inplace(
        float* __restrict__ output,
        const float* __restrict__ bias,
        int total_M
) {
    const size_t total_elements =
        (size_t)total_M * (size_t)kHiddenDim;
    const size_t stride = (size_t)gridDim.x * (size_t)kBlockThreads;
    size_t i = (size_t)blockIdx.x * (size_t)kBlockThreads
               + (size_t)threadIdx.x;
    for (; i < total_elements; i += stride) {
        const int n = (int)(i % (size_t)kHiddenDim);
        output[i] += bias[n];
    }
}

void run_cutlass_gemm(
    const float* A_ptr, const float* B_ptr, float* C_ptr,
    int M, int N, int K,
    cudaStream_t stream
) {
    CutlassGemm gemm_op;
    CutlassGemm::Arguments args(
        {M, N, K},
        {A_ptr, K},
        {B_ptr, K},
        {C_ptr, N},
        {C_ptr, N},
        {1.0f, 0.0f});
    cutlass::Status status = gemm_op(args, nullptr, stream);
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "CUTLASS fp32 GEMM (problem_037) failed: ",
        cutlassGetStatusString(status));
}

}  // namespace

void problem_037_flux_feedforward_gelu_approximate_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& fc1_weight,
    const torch::Tensor& fc1_bias,
    const torch::Tensor& fc2_weight,
    const torch::Tensor& fc2_bias,
    torch::Tensor& output,
    cudaStream_t stream
) {
    const int B = static_cast<int>(hidden_states.size(0));
    const int S = static_cast<int>(hidden_states.size(1));
    const int M = B * S;

    auto workspace = torch::empty(
        {M, kMlpHiddenDim},
        torch::TensorOptions()
            .dtype(hidden_states.dtype())
            .device(hidden_states.device()));

    const float* hidden_states_ptr = hidden_states.data_ptr<float>();
    const float* fc1_weight_ptr = fc1_weight.data_ptr<float>();
    const float* fc1_bias_ptr = fc1_bias.data_ptr<float>();
    const float* fc2_weight_ptr = fc2_weight.data_ptr<float>();
    const float* fc2_bias_ptr = fc2_bias.data_ptr<float>();
    float* workspace_ptr = workspace.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // GEMM 1: workspace = hidden_states @ fc1_weight^T.
    run_cutlass_gemm(
        hidden_states_ptr, fc1_weight_ptr, workspace_ptr,
        M, kMlpHiddenDim, kHiddenDim, stream);

    // GELU-tanh + bias in-place on workspace.
    {
        const size_t total_elements =
            (size_t)M * (size_t)kMlpHiddenDim;
        const int max_blocks = 65535;
        const int needed_blocks = static_cast<int>(
            (total_elements + kBlockThreads - 1) / kBlockThreads);
        const int num_blocks =
            needed_blocks < max_blocks ? needed_blocks : max_blocks;
        k_gelu_tanh_with_bias_inplace<<<num_blocks, kBlockThreads,
                                        0, stream>>>(
            workspace_ptr, fc1_bias_ptr, M);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "problem_037 GELU+bias kernel launch failed");
    }

    // GEMM 2: output = workspace @ fc2_weight^T.
    run_cutlass_gemm(
        workspace_ptr, fc2_weight_ptr, output_ptr,
        M, kHiddenDim, kMlpHiddenDim, stream);

    // Bias-add in-place on output.
    {
        const size_t total_elements =
            (size_t)M * (size_t)kHiddenDim;
        const int max_blocks = 65535;
        const int needed_blocks = static_cast<int>(
            (total_elements + kBlockThreads - 1) / kBlockThreads);
        const int num_blocks =
            needed_blocks < max_blocks ? needed_blocks : max_blocks;
        k_bias_add_inplace<<<num_blocks, kBlockThreads, 0, stream>>>(
            output_ptr, fc2_bias_ptr, M);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                    "problem_037 bias-add kernel launch failed");
    }
}
