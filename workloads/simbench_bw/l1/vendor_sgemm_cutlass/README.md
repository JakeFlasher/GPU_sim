# `vendor_sgemm_cutlass` FP32 successor witness

This is a project-owned direct CUTLASS successor for the frozen
`simbench_bw/vendor_sgemm_cublas` control row. It is a new implementation
identity, not a rewrite or relabeling of the frozen cuBLAS row.

The executable uses CUTLASS v4.6.1 `device::Gemm` with an sm_86 SIMT
configuration, FP32 A/B inputs, and FP32 C output. B is stored column-major for
the explicit layout contract. The source and Makefile contain no cuBLAS or
cuBLASLt include, API, or link dependency.

Build with the pinned user-space toolchain:

```sh
make -B -C workloads/simbench_bw/l1/vendor_sgemm_cutlass run-bin \
  NVCC=/usr/local/cuda-13.2/bin/nvcc \
  NVCC_CCBIN=/usr/bin/g++-14 \
  CUOBJDUMP=/usr/local/cuda-13.2/bin/cuobjdump \
  CUTLASS_DIR="$PWD/third_party/cutlass"
```

The `trace_00` 32x32x32 launch is a bounded build/smoke witness. A larger
capture shape requires independent successor-matrix and trace-budget
admission. Executable, SASS, control, source, shape, and CUTLASS provenance
must all be sealed before any hardware capture claim.
