# `vendor_gemm_cutlass` BF16 successor witness

This is a project-owned CUTLASS successor for the frozen
`simbench_bw/vendor_gemm_cublas` control row. It is not a rewrite or
relabeling of that row and does not inherit its vendor-module control contract.

The executable uses CUTLASS v4.6.1 `device::Gemm` with an sm_86 SIMT schedule,
BF16 A/B inputs, and FP32 C output. It contains no cuBLAS/cuBLASLt include,
API, or link dependency. The small `trace_00` shape is a compile/smoke witness;
a larger capture shape requires independent successor-matrix admission and a
new trace-budget decision.

Build with the pinned user-space toolchain:

```sh
make -B -C workloads/simbench_bw/l1/vendor_gemm_cutlass run-bin \
  NVCC=/usr/local/cuda-13.2/bin/nvcc \
  NVCC_CCBIN=/usr/bin/g++-14 \
  CUOBJDUMP=/usr/local/cuda-13.2/bin/cuobjdump \
  CUTLASS_DIR="$PWD/third_party/cutlass"
```

The resulting binary is evidence only after the successor witness records the
exact source, shape, CUTLASS provenance, compiler, architecture, executable,
SASS, and control-bit hashes.
