// NVFP4 linear-layer kernel pair for problem_020 on sm_120.
//
// Reference algorithm (from definition.json):
//   y[b, s, n] = sum_k (decode_w[n, k] * w_scale[n, k/16] * w_global)
//             * (decode_x[b*s, k] * x_scale[b*s, k/16] * x_global)
//
// where decode_*(nibble) follows the E2M1 lookup table
//   {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} with negative side at +8.
//
// This baseline uses the CUTLASS 4.4.1 sm_120 dense block-scaled
// NVFP4 GEMM (example
// ``79a_blackwell_geforce_nvfp4_bf16_gemm.cu`` in the local CUTLASS
// tree at ``cutlass/``) in place of a hand-written sm_120 kernel.
// The CUTLASS template handles the FP4 decode, per-K-block
// FP8 scale multiplication, and FP32 accumulation natively via the
// sm_120 block-scaled tensor cores (``OpClassBlockScaledTensorOp``).
// Combined with our existing reference-mirroring quantization prepass,
// this preserves the precision-wall invariant (per-K-block FP8 scale
// applied at the MMA boundary; FP32 accumulator; BF16 cast at output)
// AND uses the architecturally-native FP4 MMA path that plain WMMA on
// sm_120 cannot reach.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "kernel.h"

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>

namespace {

// Note: do not `using namespace cute` inside an anonymous namespace —
// cute uses internal anonymous namespaces and the cudafe stub emits
// ambiguous-reference errors. Use cute symbols qualified instead.
using cute::Shape;
using cute::_1;
using cute::_128;

// Reference _cast_to_fp4x2 threshold table (used by the pack kernel).
__device__ __forceinline__ uint8_t fp32_to_fp4_nibble(float x) {
    if (x >= 0.0f) {
        if (x <= 0.25f) return 0;
        if (x <  0.75f) return 1;
        if (x <= 1.25f) return 2;
        if (x <  1.75f) return 3;
        if (x <= 2.5f)  return 4;
        if (x <  3.5f)  return 5;
        if (x <= 5.0f)  return 6;
        return 7;
    } else {
        if (x >= -0.25f) return 8;
        if (x >  -0.75f) return 9;
        if (x >= -1.25f) return 10;
        if (x >  -1.75f) return 11;
        if (x >= -2.5f)  return 12;
        if (x >  -3.5f)  return 13;
        if (x >= -5.0f)  return 14;
        return 15;
    }
}

__global__ void fp4_pack_kernel(
    const float* __restrict__ scaled_x,
    uint8_t* __restrict__ packed,
    int M, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int K_half = K >> 1;
    const int total = M * K_half;
    if (idx >= total) return;
    const int m = idx / K_half;
    const int k_pair = idx - m * K_half;
    const int k_even = 2 * k_pair;
    const int k_odd  = k_even + 1;
    const uint8_t lo = fp32_to_fp4_nibble(scaled_x[m * K + k_even]);
    const uint8_t hi = fp32_to_fp4_nibble(scaled_x[m * K + k_odd]);
    packed[m * K_half + k_pair] = lo | (hi << 4);
}

// Remap a plain ``[D, K_blocks]`` FP8 e4m3 scale tensor (row-major,
// the shape produced by the reference's ``BlockWiseScalerNVFP4``)
// into the CUTLASS sm_120 block-scaled GEMM's interleaved layout.
// The closed-form is derived empirically from
// ``Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA`` evaluated at
// K=8192, scale group size 16. Verified offsets match the cute layout
// for (m, kb) ∈ {0..127} × {0..31}; the formula generalises to larger
// M / K_blocks via the M_outer stride.
//
// Closed form:
//   offset = (d%32)*16 + ((d/32)%4)*4 + (d/128)*(K_blocks*128)
//          + (kb%4)    + (kb/4)*K_blocks
__global__ void remap_scales_kernel(
    const uint8_t* __restrict__ plain_scales,  // [D, K_blocks] FP8 e4m3
    uint8_t* __restrict__ sfa_out,             // CUTLASS interleaved
    int D, int K_blocks) {
    const int kb = blockIdx.x * blockDim.x + threadIdx.x;
    const int d  = blockIdx.y * blockDim.y + threadIdx.y;
    if (d >= D || kb >= K_blocks) return;

    const int d_inner_a = d & 31;          // d % 32
    const int d_inner_b = (d >> 5) & 3;    // (d / 32) % 4
    const int d_outer   = d >> 7;          // d / 128
    const int kb_inner  = kb & 3;          // kb % 4
    const int kb_outer  = kb >> 2;         // kb / 4
    const int offset = d_inner_a * 16
                     + d_inner_b * 4
                     + d_outer   * (K_blocks * 128)
                     + kb_inner
                     + kb_outer  * K_blocks;
    sfa_out[offset] = plain_scales[d * K_blocks + kb];
}

// CUTLASS NVFP4 GEMM type setup — instantiation follows the
// GeForce-RTX-50-series-specific constraints (cluster 1x1x1; no
// TMA multicast; no dynamic datatypes) used in CUTLASS example
// 79a (``79a_blackwell_geforce_nvfp4_bf16_gemm.cu``).

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD   = cutlass::bfloat16_t;
using ElementC   = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag       = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128,_128,_128>;
using ClusterShape     = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

}  // namespace

void problem_020_fp4_pack_launcher(
    const torch::Tensor& scaled_x,
    torch::Tensor packed_out,
    cudaStream_t stream) {
    TORCH_CHECK(scaled_x.dim() == 2, "scaled_x must be [M, K]");
    TORCH_CHECK(packed_out.dim() == 2, "packed_out must be [M, K/2]");
    TORCH_CHECK(scaled_x.scalar_type() == at::kFloat,
                "scaled_x must be FP32");
    TORCH_CHECK(packed_out.scalar_type() == at::kByte,
                "packed_out must be uint8");
    const int M = static_cast<int>(scaled_x.size(0));
    const int K = static_cast<int>(scaled_x.size(1));
    TORCH_CHECK((K & 1) == 0, "K must be even");
    const int total = M * (K / 2);
    constexpr int kBlock = 256;
    const int grid = (total + kBlock - 1) / kBlock;
    fp4_pack_kernel<<<grid, kBlock, 0, stream>>>(
        scaled_x.data_ptr<float>(),
        packed_out.data_ptr<uint8_t>(),
        M, K);
}

void problem_020_remap_scales_launcher(
    const torch::Tensor& plain_scales,
    torch::Tensor sfa_out,
    int D,
    int K_blocks,
    cudaStream_t stream) {
    TORCH_CHECK(plain_scales.dim() == 2, "plain_scales must be 2D");
    TORCH_CHECK(plain_scales.size(0) == D && plain_scales.size(1) == K_blocks,
                "plain_scales shape mismatch");
    TORCH_CHECK(plain_scales.scalar_type() == at::kFloat8_e4m3fn,
                "plain_scales must be FP8 e4m3fn");
    TORCH_CHECK(sfa_out.scalar_type() == at::kFloat8_e4m3fn,
                "sfa_out must be FP8 e4m3fn");
    TORCH_CHECK(sfa_out.numel() >= D * K_blocks,
                "sfa_out buffer too small for D * K_blocks");
    dim3 block(32, 8, 1);
    dim3 grid((K_blocks + block.x - 1) / block.x,
              (D        + block.y - 1) / block.y, 1);
    remap_scales_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(plain_scales.data_ptr()),
        reinterpret_cast<uint8_t*>(sfa_out.data_ptr()),
        D, K_blocks);
}

void problem_020_fp4_gemm_launcher(
    const torch::Tensor& x_fp4,
    const torch::Tensor& x_sfa,
    const torch::Tensor& w_fp4,
    const torch::Tensor& w_sfb,
    torch::Tensor y_bf16,
    float global_scale,
    cudaStream_t stream) {
    TORCH_CHECK(x_fp4.dim() == 2 && w_fp4.dim() == 2,
                "fp4 inputs must be 2D");
    TORCH_CHECK(y_bf16.dim() == 2 && y_bf16.scalar_type() == at::kBFloat16,
                "y_bf16 must be 2D BF16");
    TORCH_CHECK(x_fp4.scalar_type() == at::kByte
                    && w_fp4.scalar_type() == at::kByte,
                "fp4 inputs must be uint8-packed");
    TORCH_CHECK(x_sfa.scalar_type() == at::kFloat8_e4m3fn
                    && w_sfb.scalar_type() == at::kFloat8_e4m3fn,
                "SFA/SFB must be FP8 e4m3fn");
    const int M = static_cast<int>(x_fp4.size(0));
    const int N = static_cast<int>(w_fp4.size(0));
    const int K_half = static_cast<int>(x_fp4.size(1));
    const int K = K_half * 2;
    TORCH_CHECK(w_fp4.size(1) == K_half, "w_fp4 K/2 must match x_fp4");
    TORCH_CHECK(y_bf16.size(0) == M && y_bf16.size(1) == N,
                "y_bf16 shape mismatch");
    TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16");

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<cutlass::float_e2m1_t const*>(x_fp4.data_ptr<uint8_t>()),
            stride_A,
            reinterpret_cast<cutlass::float_e2m1_t const*>(w_fp4.data_ptr<uint8_t>()),
            stride_B,
            reinterpret_cast<cutlass::float_ue4m3_t const*>(x_sfa.data_ptr()),
            layout_SFA,
            reinterpret_cast<cutlass::float_ue4m3_t const*>(w_sfb.data_ptr()),
            layout_SFB,
        },
        {
            {global_scale, 0.0f},
            nullptr,
            stride_C,
            reinterpret_cast<cutlass::bfloat16_t*>(y_bf16.data_ptr()),
            stride_D,
        }
    };

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    auto workspace = at::empty(
        {static_cast<int64_t>(workspace_size)},
        x_fp4.options().dtype(at::kByte));
    cutlass::Status status = gemm.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS NVFP4 GEMM can_implement failed: ",
                cutlass::cutlassGetStatusString(status));
    status = gemm.initialize(arguments, workspace.data_ptr<uint8_t>(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS NVFP4 GEMM initialize failed: ",
                cutlass::cutlassGetStatusString(status));
    status = gemm.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS NVFP4 GEMM run failed: ",
                cutlass::cutlassGetStatusString(status));
}
