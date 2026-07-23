# `vendor_sgemm_strided_batched_cutlass` successor witness

This is a project-owned direct CUTLASS v4.6.1 successor for the frozen
`vendor_sgemm_strided_batched_cublas` control row. It has a distinct namespace
and does not rewrite, relabel, or inherit the frozen vendor-module contract.

The implementation uses `cutlass::gemm::device::GemmBatched` (not the classic
non-batched `device::Gemm`) with FP32 row-major A, FP32 column-major B, and FP32
row-major C/D. Strides are explicit element strides (`M*K`, `K*N`, and `M*N`)
and `batch_count` is passed to the CUTLASS Arguments object. The source and
Makefile contain no vendor-library include, API, or link dependency.

Build with the pinned user-space toolchain:

```sh
make -B -C workloads/simbench_bw/l1/vendor_sgemm_strided_batched_cutlass run-bin \
  NVCC=/usr/local/cuda-13.2/bin/nvcc \
  NVCC_CCBIN=/usr/bin/g++-14 \
  CUOBJDUMP=/usr/local/cuda-13.2/bin/cuobjdump \
  CUTLASS_DIR="$PWD/third_party/cutlass"
```

The bounded `trace_00` launch is 32x32x32 with batch_count 4. The witness
emits batch zero for the correctness payload while sealing the full batched
launch contract; `output_scope=batch_zero_only` makes that bounded payload
explicit. Hardware capture admission requires an independent matrix row,
launch inventory, trace budget, and matched label/trace evidence.
