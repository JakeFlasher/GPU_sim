/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// CUDA C ABI bridge for SOL's 1x128 activation / 128x128 weight FP8 scales.
// Based on CUTLASS GeForce example 87b. Outputs FP32 so bias is applied before
// the workload's BF16 cast, as in the supplied SOL reference.
#include <cstdio>
#include <exception>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;
using FP8 = cutlass::float_e4m3_t;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using MmaTile = Shape<_128,_128,_128>;
using Cluster = Shape<_1,_1,_1>;
using Scales = cutlass::detail::Sm120BlockwiseScaleConfig<1,128,128>;
using SFA = decltype(Scales::deduce_layoutSFA());
using SFB = decltype(Scales::deduce_layoutSFB());
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, MmaTile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, float, Row, 4, float, Row, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    FP8, cute::tuple<Row,SFA>, 16, FP8, cute::tuple<Col,SFB>, 16,
    float, MmaTile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,
    cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
static thread_local char error_text[512];
extern "C" const char *sm120_fp8_error() { return error_text; }

// A [M,K], B [N,K] (both K-contiguous), SA [K/128,M], SB [K/128,N/128].
// The transposed scale buffers are MN-major, exactly the 87b ScaleConfig ABI.
extern "C" int sm120_fp8_groupwise(const void *A, const void *B,
    const float *SA, const float *SB, float *D, int M, int N, int K,
    void *stream_ptr) {
  error_text[0] = 0;
  if (!A || !B || !SA || !SB || !D || M <= 0 || N <= 0 || K <= 0 || N%128 || K%128) {
    snprintf(error_text, sizeof(error_text), "require M>0, N and K positive multiples of 128, and device pointers");
    return 1;
  }
  void *workspace = nullptr;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto fail_cutlass = [&](const char *where, cutlass::Status status) {
    snprintf(error_text, sizeof(error_text), "%s: %s", where, cutlassGetStatusString(status));
    if (workspace) cudaFree(workspace);
    return 2;
  };
  try {
    auto shape = make_shape(M,N,K,1);
    auto da = cutlass::make_cute_packed_stride(typename Kernel::StrideA{},make_shape(M,K,1));
    auto db = cutlass::make_cute_packed_stride(typename Kernel::StrideB{},make_shape(N,K,1));
    auto dc = cutlass::make_cute_packed_stride(typename Kernel::StrideC{},make_shape(M,N,1));
    auto dd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{},make_shape(M,N,1));
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
      {static_cast<const FP8*>(A),da,static_cast<const FP8*>(B),db,
       SA,Scales::tile_atom_to_shape_SFA(shape),SB,Scales::tile_atom_to_shape_SFB(shape)},
      {{}, nullptr,dc, D,dd}};
    args.epilogue.thread.alpha=1.0f;
    args.epilogue.thread.beta=0.0f;
    Gemm gemm;
    auto status=gemm.can_implement(args);
    if(status!=cutlass::Status::kSuccess) return fail_cutlass("can_implement",status);
    size_t bytes=Gemm::get_workspace_size(args);
    if(bytes) {
      auto err=cudaMalloc(&workspace,bytes);
      if(err!=cudaSuccess) {
        snprintf(error_text,sizeof(error_text),"workspace: %s",cudaGetErrorString(err)); return 3;
      }
    }
    status=gemm.initialize(args,workspace,stream);
    if(status!=cutlass::Status::kSuccess) return fail_cutlass("initialize",status);
    status=gemm.run(stream);
    if(status!=cutlass::Status::kSuccess) return fail_cutlass("run",status);
    // Keep Python-owned operands and the per-call workspace alive until done.
    auto err=cudaStreamSynchronize(stream);
    if(workspace) cudaFree(workspace);
    if(err!=cudaSuccess) {
      snprintf(error_text,sizeof(error_text),"completion: %s",cudaGetErrorString(err)); return 4;
    }
    return 0;
  } catch(const std::exception &e) {
    if(workspace) cudaFree(workspace);
    snprintf(error_text,sizeof(error_text),"exception: %s",e.what()); return 5;
  }
}
