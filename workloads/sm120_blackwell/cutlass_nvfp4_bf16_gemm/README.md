# CUTLASS SM120 NVFP4→BF16 GEMM

This is a GeForce-specific SM120 workload copied from NVIDIA CUTLASS 4.8,
commit `0b55a2f691d69981583568fd9eb69687b1f0de8a`, example
`79a_blackwell_geforce_nvfp4_bf16_gemm.cu`.

It uses `cutlass::arch::Sm120`, `OpClassBlockScaledTensorOp`, warp-level
block-scaled MMA, and a 1×1×1 cluster. It is intentionally separate from the
portable `workloads/simbench_bw` rows and must not be built for SM86 or SM100.

Build uses CUDA 13.2 and `/usr/bin/g++-14` explicitly:

```bash
make CUDA_INSTALL_PATH=/usr/local/cuda-13.2 CXX=/usr/bin/g++-14 \
  CUTLASS_ROOT=/root/GPU_simulator_rebase/third_party/cutlass
```

The source is a production-style CUTLASS control for the custom NVFP4
microbenchmarks; it is not a simulator implementation.
